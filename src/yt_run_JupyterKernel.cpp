#include "libyt.h"
#include "yt_combo.h"

#if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL)
#include <fstream>
#include <iostream>
#include <xeus-zmq/xserver_zmq.hpp>
#include <xeus/xkernel.hpp>
#include <xeus/xkernel_configuration.hpp>

#include "LibytProcessControl.h"
#include "libyt_kernel.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_run_JupyterKernel
// Description :  Start libyt kernel for Jupyter Notebook access
//
// Note        :  1. Must enable -DINTERACTIVE_MODE and -DJUPYTER_KERNEL.
//                2. Must install libyt provisioner for jupyter client.
//                3. This API is like interactive mode, but with Jupyter Notebook access for better UI.
//                4. This API forces kernel to be on MPI process 0 (root).
//                5. Simulation + libyt processes and Jupyter server are launch in separate process.
//                6. Currently, this API is tested on local computing resources. Need further improvement
//                   when libyt kernel is launch on remote nodes.
//
// Parameter   :  const char *flag_file_name       : once this file is detected, it will activate libyt kernel.
//                bool        use_connection_file  : use connection file set by user
//                const char *connection_file_name : API will read this file if use_connection_file is true
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_JupyterKernel(const char* flag_file_name, bool use_connection_file, const char* connection_file_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

#if !defined(INTERACTIVE_MODE) || !defined(JUPYTER_KERNEL)
    log_error(
        "Cannot start libyt kernel for Jupyter. Please compile libyt with -DINTERACTIVE_MODE and -DJUPYTER_KERNEL.\n");
    return YT_FAIL;
#else
    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // run new added functions
    if (g_func_status_list.run_func() != YT_SUCCESS) {
        YT_ABORT("Something went wrong when running new added functions\n");
    }

    // TODO: (LATER) see if we need to start libyt kernel by checking if file flag_file_name exist.

    // Launch libyt kernel on root process
    int root = 0;
    if (g_myrank == root) {
        // Get root process PID
        std::ofstream file;
        file.open("libyt_kernel_pid.txt", std::ios::out | std::ios::trunc);
        file << getpid();
        file.close();

        // Make context and create libyt kernel
        auto context = xeus::make_context<zmq::context_t>();
        std::unique_ptr<LibytKernel> interpreter = std::make_unique<LibytKernel>();

        if (use_connection_file) {
            // Load configuration (ex: port, ip, ...)
            std::string config_file = std::string(connection_file_name);
            xeus::xconfiguration config = xeus::load_configuration(config_file);

            xeus::xkernel libyt_kernel(config, xeus::get_user_name(), std::move(context), std::move(interpreter),
                                       xeus::make_xserver_zmq);

            // Launch kernel
            log_info("Launching libyt kernel using provided connection file \"%s\" ...\n", connection_file_name);
            libyt_kernel.start();
        } else {
            xeus::xkernel libyt_kernel(xeus::get_user_name(), std::move(context), std::move(interpreter),
                                       xeus::make_xserver_zmq);

            // Output connection info
            const auto& config = libyt_kernel.get_config();
            file.open("libyt_kernel_connection.json", std::ios::out | std::ios::trunc);
            file << "{\n";
            file << "    \"transport\": \"" + config.m_transport + "\",\n";
            file << "    \"ip\": \"" + config.m_ip + "\",\n";
            file << "    \"control_port\": " + config.m_control_port + ",\n";
            file << "    \"shell_port\": " + config.m_shell_port + ",\n";
            file << "    \"stdin_port\": " + config.m_stdin_port + ",\n";
            file << "    \"iopub_port\": " + config.m_iopub_port + ",\n";
            file << "    \"hb_port\": " + config.m_hb_port + ",\n";
            file << "    \"signature_scheme\": \"" + config.m_signature_scheme + "\",\n";
            file << "    \"key\": \"" + config.m_key + "\"\n";
            file << "}\n";
            file.close();

            // Launch kernel
            log_info("Launching libyt kernel, connection info are stored in \"libyt_kernel_connection.json\" ...\n");
            libyt_kernel.start();
        }
    }
#ifndef SERIAL_MODE
    // TODO: Launch worker on non-root ranks
    else {
    }
#endif  // #ifndef SERIAL_MODE

    return YT_SUCCESS;
#endif  // #if !defined(INTERACTIVE_MODE) && !defined(JUPYTER_KERNEL)
}
