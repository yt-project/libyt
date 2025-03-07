#include "libyt.h"
#include "logging.h"
#include "timer.h"

#ifdef JUPYTER_KERNEL
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <xeus-zmq/xserver_zmq.hpp>
#include <xeus/xkernel.hpp>
#include <xeus/xkernel_configuration.hpp>

#include "function_info.h"
#include "libyt_kernel.h"
#include "libyt_process_control.h"
#include "libyt_utilities.h"
#ifndef SERIAL_MODE
#include "libyt_worker.h"
#endif
#endif

/**
 * \defgroup api_yt_run_JupyterKernel libyt API: yt_run_JupyterKernel
 * \fn int yt_run_JupyterKernel(const char* flag_file_name, bool use_connection_file)
 * \brief Start libyt kernel for Jupyter Notebook access
 * \details
 * 1. This API provides an access point for Jupyter Notebook.
 * 2. This API forces kernel to be on MPI process 0 (root).
 * 3. Jupyter Notebook/JupyterLab are launched in separated process.
 *
 * \rst
 * .. important::
 *    Connection file must be ``libyt_kernel_connection.json`` if
 *    ``use_connection_file`` is true.
 *
 * .. note::
 *    This API is only available when libyt is compiled with -DJUPYTER_KERNEL=ON.
 * \endrst
 *
 * @param flag_file_name[in] Flag file name exists means it will activate libyt kernel
 * @param use_connection_file[in] Use connection file set by user or not
 * @return
 */
int yt_run_JupyterKernel(const char* flag_file_name, bool use_connection_file) {
  SET_TIMER(__PRETTY_FUNCTION__);

#ifndef JUPYTER_KERNEL
  logging::LogError("Cannot start libyt kernel for Jupyter. Please compile libyt with "
                    "-DJUPYTER_KERNEL.\n");
  return YT_FAIL;
#else
  // check if libyt has been initialized
  if (!LibytProcessControl::Get().libyt_initialized_) {
    YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
  }

  // run new added functions
  LibytProcessControl::Get().function_info_list_.RunEveryFunction();

  // see if we need to start libyt kernel by checking if file flag_file_name exist.
  if (libyt_utilities::DoesFileExist(flag_file_name)) {
    logging::LogInfo(
        "File '%s' detected, preparing libyt kernel for Jupyter Notebook access ...\n",
        flag_file_name);
  } else {
    logging::LogInfo("No file '%s' detected, skip starting libyt kernel for Jupyter "
                     "Notebook access ...\n",
                     flag_file_name);
    return YT_SUCCESS;
  }

#ifndef SERIAL_MODE
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  // Basic libyt kernel info
  const char* kernel_pid_filename = "libyt_kernel_pid.txt";
  const char* kernel_connection_filename = "libyt_kernel_connection.json";

  int mpi_rank = LibytProcessControl::Get().mpi_rank_;
  int mpi_root = LibytProcessControl::Get().mpi_root_;
  int mpi_size = LibytProcessControl::Get().mpi_size_;

  // Launch libyt kernel on root process
  if (mpi_rank == mpi_root) {
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
          if (!libyt_utilities::DoesFileExist(kernel_connection_filename)) {
            throw -1;
          }

          // Load configuration (ex: port, ip, ...)
          xeus::xconfiguration config =
              xeus::load_configuration(std::string(kernel_connection_filename));

          // Make context and create libyt kernel
          auto context = xeus::make_context<zmq::context_t>();
          std::unique_ptr<LibytKernel> interpreter = std::make_unique<LibytKernel>();

          libyt_kernel_ptr = new xeus::xkernel(config,
                                               xeus::get_user_name(),
                                               std::move(context),
                                               std::move(interpreter),
                                               xeus::make_xserver_zmq);

          complete = true;
        } catch (int err_code) {
          if (err_code == -1) {
            logging::LogError("Unable to find \"%s\" ...\n", kernel_connection_filename);
          }
        } catch (const nlohmann::json::parse_error& e) {
          switch (e.id) {
            case 101: {
              logging::LogError("Unable to parse \"%s\". This error may be caused by not "
                                "enclosing key-value pairs in "
                                "{} bracket or not separating key-value pairs using ',' "
                                "(nlohmann json err "
                                "msg: %s)\n",
                                kernel_connection_filename,
                                e.what());
              break;
            }
            default: {
              logging::LogError("Unable to parse \"%s\" (nlohmann json err msg: %s)\n",
                                kernel_connection_filename,
                                e.what());
            }
          }
        } catch (const nlohmann::json::type_error& e) {
          switch (e.id) {
            case 302: {
              logging::LogError("Error occurred while reading keys in \"%s\". "
                                "This error may be caused by missing one of the keys "
                                "(\"transport\", \"ip\", "
                                "\"control_port\", \"shell_port\", \"stdin_port\", "
                                "\"iopub_port\", \"hb_port\", "
                                "\"signature_scheme\", \"key\") "
                                "(nlohmann json err msg: %s)\n",
                                kernel_connection_filename,
                                e.what());
              break;
            }
            default: {
              logging::LogError("Error occurred while reading keys in \"%s\" (nlohmann "
                                "json err msg: %s)\n",
                                kernel_connection_filename,
                                e.what());
            }
          }
        } catch (const nlohmann::json::exception& e) {
          logging::LogError(
              "Other errors occurred when reading \"%s\" (nlohmann json err msg: %s)\n",
              kernel_connection_filename,
              e.what());
        } catch (const std::out_of_range& e) {
          logging::LogError("This error may be caused by not providing "
                            "\"signature_scheme\" and \"key\" in \"%s\" "
                            "(std::string err msg: %s)\n",
                            kernel_connection_filename,
                            e.what());
        } catch (const zmq::error_t& e) {
          logging::LogError("Port address already in use, please change port number (zmq "
                            "err msg: %s)\n",
                            e.what());
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }

      // Launch kernel
      logging::LogInfo(
          "Launching libyt kernel using provided connection file \"%s\" ...\n",
          kernel_connection_filename);
      if (libyt_kernel_ptr != nullptr) {
        libyt_kernel_ptr->start();
      } else {
        logging::LogInfo(
            "Launching libyt kernel using provided connection file \"%s\" ... failed\n",
            kernel_connection_filename);
      }
      delete libyt_kernel_ptr;
    } else {
      // Make context and create libyt kernel
      auto context = xeus::make_context<zmq::context_t>();
      std::unique_ptr<LibytKernel> interpreter = std::make_unique<LibytKernel>();

      xeus::xkernel libyt_kernel(xeus::get_user_name(),
                                 std::move(context),
                                 std::move(interpreter),
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
      logging::LogInfo(
          "Launching libyt kernel, connection info are stored in \"%s\" ...\n",
          kernel_connection_filename);
      libyt_kernel.start();
    }

    // Remove libyt_kernel_pid.txt file when kernel is shut down.
    std::remove(kernel_pid_filename);
  }
#ifndef SERIAL_MODE
  else {
    LibytWorker libyt_worker(mpi_rank, mpi_size, mpi_root);
    libyt_worker.start();
  }
#endif

#ifndef SERIAL_MODE
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  return YT_SUCCESS;
#endif  // #ifndef JUPYTER_KERNEL
}
