/*****************************************************************************\
 *  mpi_torchrun.c - PyTorch torchrun plugin
 *****************************************************************************
 *  Copyright 2024 Mila - Institut québécois d'intelligence artificielle
 *  Written by Olexa Bilaniuk <olexa.bilaniuk@mila.quebec>
 *
 *  This file is part of Slurm, a resource management program.
 *  For details, see <https://slurm.schedmd.com/>.
 *  Please also read the included file: DISCLAIMER.
 *
 *  Slurm is free software; you can redistribute it and/or modify it under
 *  the terms of the GNU General Public License as published by the Free
 *  Software Foundation; either version 2 of the License, or (at your option)
 *  any later version.
 *
 *  In addition, as a special exception, the copyright holders give permission
 *  to link the code of portions of this program with the OpenSSL library under
 *  certain conditions as described in each individual source file, and
 *  distribute linked combinations including the two. You must obey the GNU
 *  General Public License in all respects for all of the code used other than
 *  OpenSSL. If you modify file(s) with this exception, you may extend this
 *  exception to your version of the file(s), but you are not obligated to do
 *  so. If you do not wish to do so, delete this exception statement from your
 *  version.  If you delete this exception statement from all source files in
 *  the program, then also delete it here.
 *
 *  Slurm is distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 *  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 *  details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with Slurm; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA.
\*****************************************************************************/

#include "config.h"

#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#ifdef HAVE_GETRANDOM
#include <sys/random.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "slurm/slurm_errno.h"
#include "src/common/slurm_xlator.h"

#include "src/common/env.h"
#include "src/common/hostlist.h"
#include "src/common/read_config.h"
#include "src/common/slurm_protocol_api.h"
#include "src/common/xstring.h"
#include "src/interfaces/mpi.h"
#include "src/slurmd/slurmstepd/slurmstepd_job.h"

/*
 * These variables are required by the generic plugin interface.  If they
 * are not found in the plugin, the plugin loader will ignore it.
 *
 * plugin_name - a string giving a human-readable description of the
 * plugin.  There is no maximum length, but the symbol must refer to
 * a valid string.
 *
 * plugin_type - a string suggesting the type of the plugin or its
 * applicability to a particular form of data or method of data handling.
 * If the low-level plugin API is used, the contents of this string are
 * unimportant and may be anything.  Slurm uses the higher-level plugin
 * interface which requires this string to be of the form
 *
 *      <application>/<method>
 *
 * where <application> is a description of the intended application of
 * the plugin (e.g., "switch" for Slurm switch) and <method> is a description
 * of how this plugin satisfies that application.  Slurm will only load
 * a switch plugin if the plugin_type string has a prefix of "switch/".
 *
 * plugin_version - an unsigned 32-bit integer containing the Slurm version
 * (major.minor.micro combined into a single number).
 */
const char plugin_name[] = "mpi torchrun plugin";
const char plugin_type[] = "mpi/torchrun";
const uint32_t plugin_id = MPI_PLUGIN_TORCHRUN;
const uint32_t plugin_version = SLURM_VERSION_NUMBER;

#define TORCHRUN_DEFAULT_PORT 29400

extern int mpi_p_slurmstepd_prefork(const stepd_step_rec_t *step, char ***env)
{
	return SLURM_SUCCESS;
}

extern int mpi_p_slurmstepd_task(const mpi_task_info_t *mpi_task, char ***env)
{
	/*
	 * Set environment variables.
	 *
	 * There exists documentary evidence in PyTorch Distributed [1] and
	 * Elastic [2] for four primary environment variables:
	 *
	 *   - MASTER_ADDR (set in mpi_p_client_prelaunch())
	 *   - MASTER_PORT (set in mpi_p_client_prelaunch())
	 *   - RANK
	 *   - WORLD_SIZE
	 *
	 * as well as a few additional ones specific to PyTorch Elastic that we
	 * choose to set as well:
	 *
	 *   - LOCAL_RANK
	 *   - GROUP_RANK
	 *   - LOCAL_WORLD_SIZE
	 *
	 * As Slurm is inelastic (a failed task is not restarted), and neither
	 * this module nor PyTorch Elastic itself support heterogeneous
	 * layouts, we do not set the following:
	 *
	 *   - ROLE_RANK
	 *   - ROLE_WORLD_SIZE
	 *   - TORCHELASTIC_RESTART_COUNT
	 *   - TORCHELASTIC_MAX_RESTARTS
	 *   - TORCHELASTIC_RUN_ID
	 *
	 * By assumption,  the number of nodes can be calculated anywhere as
	 *
	 *     NUM_NODES = WORLD_SIZE / LOCAL_WORLD_SIZE
	 *
	 * but there is no evidence that a standardized environment variable
	 * holds this value in PyTorch contexts.
	 *
	 *
	 * BIBLIOGRAPHY
	 *
	 * [1] https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
	 * [2] https://pytorch.org/docs/master/elastic/run.html#environment-variables
	 */

	env_array_overwrite_fmt(env, "RANK", "%u", mpi_task->gtaskid);
	env_array_overwrite_fmt(env, "WORLD_SIZE", "%u", mpi_task->ntasks);
	env_array_overwrite_fmt(env, "LOCAL_RANK", "%u", mpi_task->ltaskid);
	env_array_overwrite_fmt(env, "GROUP_RANK", "%u", mpi_task->nodeid);
	env_array_overwrite_fmt(env, "LOCAL_WORLD_SIZE", "%u",
				mpi_task->ltasks);

	return SLURM_SUCCESS;
}

extern mpi_plugin_client_state_t *
mpi_p_client_prelaunch(const mpi_step_info_t *mpi_step, char ***env)
{
	uint32_t n = 0, i = 0;
	char* node0_name = NULL;
	char node0_addrstr[128] = {0};
	slurm_addr_t node0_addr;
	int node0_fd = -1;
	unsigned int node0_port = 0;

	/*
	 * The following is copied from mpi/cray_shasta. It was felt that a
	 * shared secret value might genuinely be useful for distributed
	 * PyTorch programs as well.
	 */
#ifdef HAVE_GETRANDOM
#define PMI_SHARED_SECRET_ENV "PMI_SHARED_SECRET"
	static uint64_t shared_secret = 0;
	static pthread_mutex_t shared_secret_mutex = PTHREAD_MUTEX_INITIALIZER;

	slurm_mutex_lock(&shared_secret_mutex);

	/*
	 * Get a non-zero pseudo-random value. getrandom() is guaranteed to
	 * return up to 256 bytes uninturrupted. The only error we might expect
	 * here is ENOSYS, indicating that the kernel does not implement the
	 * getrandom() system call. getrandom() should be present on all
	 * supported systems.
	 */

	if (!shared_secret &&
	    getrandom(&shared_secret, sizeof(shared_secret), 0) < 0) {
		error("%s: getrandom() failed: %m", __func__);
		slurm_mutex_unlock(&shared_secret_mutex);
		return NULL;
	}

	/* Set PMI_SHARED_SECRET for PMI authentication */
	env_array_overwrite_fmt(env, PMI_SHARED_SECRET_ENV, "%"PRIu64,
				shared_secret);

	slurm_mutex_unlock(&shared_secret_mutex);
#endif

	/*
	 * If user manually configured MASTER_ADDR or MASTER_PORT, assume set
	 * correctly and do not override. The responsibility is no longer ours.
	 */
	if(getenvp(*env, "MASTER_ADDR") ||
	   getenvp(*env, "MASTER_PORT"))
		return (void *)0xdeadbeef; /* only return NULL on error */

	/*
	 * Otherwise, set MASTER_ADDR and MASTER_PORT to computed values.
	 * Set MASTER_ADDR as the node to which global task ID 0 was assigned.
	 */
	for (n=0; n < mpi_step->step_layout->node_cnt; n++)
		for (i=0; i < mpi_step->step_layout->tasks[n]; i++)
			if (mpi_step->step_layout->tids[n][i] == 0)
				goto found_node0;

	error("%s: no node has task id 0! %m", __func__);
	return NULL;

	found_node0:
	node0_name = nodelist_nth_host(mpi_step->step_layout->node_list, n);
	if (!node0_name) {
		error("%s: Could not determine task id 0's node name: %m",
			__func__);
		return NULL;
	}
	slurm_set_addr(&node0_addr, slurm_conf.slurmd_port, node0_name);
	free(node0_name);
	node0_name = NULL;

	/**
	 * Contact master node for a new TCP port number, then retrieve remote
	 * address and port number. Close socket immediately afterwards. In the
	 * event of failure, pick port 29400 as a last gasp, matching PyTorch
	 * Elastic's documented default.
	 *
	 * Yes, extremely cheesy. Would be cleaner to do an RPC call of some
	 * sort or use a Slurm API for this.
	 */
	node0_fd = slurm_open_msg_conn(&node0_addr);
	if (node0_fd < 0) {
		error("%s: Could not connect to task id 0's node: %m",
			__func__);
		return NULL;
	}
	slurm_get_peer_addr(node0_fd, &node0_addr);
	slurm_get_ip_str(&node0_addr, node0_addrstr, sizeof(node0_addrstr));
	node0_port = slurm_get_port(&node0_addr);
	node0_port = node0_port ? node0_port : TORCHRUN_DEFAULT_PORT;
	close(node0_fd);

	/* Assign computed values to environment variables. */
	env_array_overwrite_fmt(env, "MASTER_ADDR", "%s", node0_addrstr);
	env_array_overwrite_fmt(env, "MASTER_PORT", "%u", node0_port);

	/* only return NULL on error */
	return (void *)0xdeadbeef;
}

extern int mpi_p_client_fini(mpi_plugin_client_state_t *state)
{
	return SLURM_SUCCESS;
}

extern int init(void)
{
	return SLURM_SUCCESS;
}

extern int fini(void)
{
	return SLURM_SUCCESS;
}

extern void mpi_p_conf_options(s_p_options_t **full_options,
			       int *full_opt_cnt)
{
}

extern void mpi_p_conf_set(s_p_hashtbl_t *tbl)
{
}

extern s_p_hashtbl_t *mpi_p_conf_get(void)
{
	return NULL;
}

extern List mpi_p_conf_get_printable(void)
{
	return NULL;
}
