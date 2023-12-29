#include "libyt.h"
#include "yt_combo.h"

#if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL)
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <xeus-zmq/xserver_zmq.hpp>
#include <xeus/xkernel.hpp>
#include <xeus/xkernel_configuration.hpp>

#include "LibytProcessControl.h"
#include "libyt_kernel.h"
#ifndef SERIAL_MODE
#include "libyt_worker.h"
#endif
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_run_JupyterKernel
// Description :  Start libyt kernel for Jupyter Notebook access
//
// Notes       :  1. Must enable -DINTERACTIVE_MODE and -DJUPYTER_KERNEL.
//                2. Must install jupyter_libyt for jupyter client.
//                3. This API is like interactive mode, but with Jupyter Notebook access for better UI.
//                4. This API forces kernel to be on MPI process 0 (root).
//                5. Simulation + libyt processes and Jupyter server are launch in separate process.
//                6. Currently, this API is tested on local computing resources. Need further improvement
//                   when libyt kernel is launch on remote nodes.
//                7. Connection file must be "libyt_kernel_connection.json" is use_connection_file = true.
//
// Parameter   :  const char *flag_file_name       : once this file is detected, it will activate libyt kernel.
//                bool        use_connection_file  : use connection file set by user
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_JupyterKernel(const char* flag_file_name, bool use_connection_file) {
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

    // see if we need to start libyt kernel by checking if file flag_file_name exist.
    struct stat buffer;
    if (stat(flag_file_name, &buffer) != 0) {
        log_info("No file '%s' detected, skip starting libyt kernel for Jupyter Notebook access ...\n", flag_file_name);
        return YT_SUCCESS;
    } else {
        log_info("File '%s' detected, preparing libyt kernel for Jupyter Notebook access ...\n", flag_file_name);
    }

#ifndef SERIAL_MODE
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    // Basic libyt kernel info
    const char* kernel_pid_filename = "libyt_kernel_pid.txt";
    const char* kernel_connection_filename = "libyt_kernel_connection.json";

    // Launch libyt kernel on root process
    if (g_myrank == g_myroot) {
        // Get root process PID
        std::ofstream file;
        file.open(kernel_pid_filename, std::ios::out | std::ios::trunc);
        file << getpid();
        file.close();

        if (use_connection_file) {
            // To prevent process abort because of faulty connection file error,
            // put loading connection and create kernel in try and catch in while loop
            xeus::xkernel* libyt_kernel_ptr = nullptr;

            bool complete = false;
            while (!complete) {
                try {
                    // Check if connection file exist
                    if (stat(kernel_connection_filename, &buffer) != 0) {
                        throw -1;
                    }

                    // Load configuration (ex: port, ip, ...)
                    xeus::xconfiguration config = xeus::load_configuration(std::string(kernel_connection_filename));

                    // Make context and create libyt kernel
                    auto context = xeus::make_context<zmq::context_t>();
                    std::unique_ptr<LibytKernel> interpreter = std::make_unique<LibytKernel>();

                    libyt_kernel_ptr = new xeus::xkernel(config, xeus::get_user_name(), std::move(context),
                                                         std::move(interpreter), xeus::make_xserver_zmq);

                    complete = true;
                } catch (int err_code) {
                    if (err_code == -1) {
                        log_error("Unable to find \"%s\" ...\n", kernel_connection_filename);
                    }
                } catch (const nlohmann::json::parse_error& e) {
                    switch (e.id) {
                        case 101: {
                            log_error(
                                "Unable to parse \"%s\". This error may be caused by not enclosing key-value pairs in "
                                "{} bracket or not separating key-value pairs using ',' (nlohmann json err "
                                "msg: %s)\n",
                                kernel_connection_filename, e.what());
                            break;
                        }
                        default: {
                            log_error("Unable to parse \"%s\" (nlohmann json err msg: %s)\n",
                                      kernel_connection_filename, e.what());
                        }
                    }
                } catch (const nlohmann::json::type_error& e) {
                    switch (e.id) {
                        case 302: {
                            log_error("Error occurred while reading keys in \"%s\". "
                                      "This error may be caused by missing one of the keys (\"transport\", \"ip\", "
                                      "\"control_port\", \"shell_port\", \"stdin_port\", \"iopub_port\", \"hb_port\", "
                                      "\"signature_scheme\", \"key\") "
                                      "(nlohmann json err msg: %s)\n",
                                      kernel_connection_filename, e.what());
                            break;
                        }
                        default: {
                            log_error("Error occurred while reading keys in \"%s\" (nlohmann json err msg: %s)\n",
                                      kernel_connection_filename, e.what());
                        }
                    }
                } catch (const nlohmann::json::exception& e) {
                    log_error("Other errors occurred when reading \"%s\" (nlohmann json err msg: %s)\n",
                              kernel_connection_filename, e.what());
                } catch (const std::out_of_range& e) {
                    log_error("This error may be caused by not providing \"signature_scheme\" and \"key\" in \"%s\" "
                              "(std::string err msg: %s)\n",
                              kernel_connection_filename, e.what());
                } catch (const zmq::error_t& e) {
                    log_error("Port address already in use, please change port number (zmq err msg: %s)\n", e.what());
                }
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }

            // Launch kernel
            log_info("Launching libyt kernel using provided connection file \"%s\" ...\n", kernel_connection_filename);
            if (libyt_kernel_ptr != nullptr) {
                libyt_kernel_ptr->start();
            } else {
                log_info("Launching libyt kernel using provided connection file \"%s\" ... failed\n",
                         kernel_connection_filename);
            }
            delete libyt_kernel_ptr;
        } else {
            // Make context and create libyt kernel
            auto context = xeus::make_context<zmq::context_t>();
            std::unique_ptr<LibytKernel> interpreter = std::make_unique<LibytKernel>();

            xeus::xkernel libyt_kernel(xeus::get_user_name(), std::move(context), std::move(interpreter),
                                       xeus::make_xserver_zmq);

            // Output connection info
            const auto& config = libyt_kernel.get_config();
            file.open(kernel_connection_filename, std::ios::out | std::ios::trunc);
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
            log_info("Launching libyt kernel, connection info are stored in \"%s\" ...\n", kernel_connection_filename);
            libyt_kernel.start();
        }

        // Remove libyt_kernel_pid.txt file when kernel is shut down.
        std::remove(kernel_pid_filename);
    }
#ifndef SERIAL_MODE
    else {
        LibytWorker libyt_worker(g_myrank, g_mysize, g_myroot);
        libyt_worker.start();
    }
#endif

#ifndef SERIAL_MODE
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    return YT_SUCCESS;
#endif  // #if !defined(INTERACTIVE_MODE) && !defined(JUPYTER_KERNEL)
}
