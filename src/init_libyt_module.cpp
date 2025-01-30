#include <cstring>
#include <fstream>
#include <string>

#include "function_info.h"
#include "libyt_process_control.h"
#include "libyt_utilities.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
// #include "pybind11/embed.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  init_libyt_module
// Description :  Initialize the libyt module
//
// Note        :  1. Import newly created libyt module.
//                2. Load user script to python.
//                3. In INTERACTIVE_MODE:
//                   (1) libyt.interactive_mode["script_globals"] = sys.modules["<script>"].__dict__
//                   (2) libyt.interactive_mode["func_err_msg"] = dict()
//                4. Bind py_grid_data/py_particle_data/py_hierarchy/py_param_yt/py_param_user/
//                   This is only needed in Pybind11
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int InitLibytModule() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // import newly created libyt module
    if (PyRun_SimpleString("import libyt\n") == 0)
        log_debug("Import libyt module ... done\n");
    else
        YT_ABORT("Import libyt module ... failed!\n");

#ifdef USE_PYBIND11
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    LibytProcessControl::Get().data_structure_amr_.SetPythonBindings(
        libyt.attr("hierarchy").ptr(), libyt.attr("grid_data").ptr(), libyt.attr("particle_data").ptr());
    LibytProcessControl::Get().py_param_yt_ = libyt.attr("param_yt").ptr();
    LibytProcessControl::Get().py_param_user_ = libyt.attr("param_user").ptr();
    LibytProcessControl::Get().py_libyt_info_ = libyt.attr("libyt_info").ptr();
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    LibytProcessControl::Get().py_interactive_mode_ = libyt.attr("interactive_mode").ptr();
#endif
#endif

    // check if script exist
    if (LibytProcessControl::Get().mpi_rank_ == 0) {
        std::string script_fullname = std::string(LibytProcessControl::Get().param_libyt_.script) + std::string(".py");
        if (libyt_utilities::DoesFileExist(script_fullname.c_str())) {
            log_debug("Finding user script %s ... done\n", script_fullname.c_str());
        } else {
            log_info("Unable to find user script %s, creating one ...\n", script_fullname.c_str());
            std::ofstream python_script(script_fullname.c_str());
            python_script.close();
            log_info("Creating empty user script %s ... done\n", script_fullname.c_str());
        }
    }

#ifndef SERIAL_MODE
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // import YT inline analysis script
    int command_width = 8 + strlen(LibytProcessControl::Get().param_libyt_.script);  // 8 = "import " + '\0'
    char* command = (char*)malloc(command_width * sizeof(char));
    sprintf(command, "import %s", LibytProcessControl::Get().param_libyt_.script);

    if (PyRun_SimpleString(command) == 0)
        log_info("Importing YT inline analysis script \"%s\" ... done\n",
                 LibytProcessControl::Get().param_libyt_.script);
    else {
        free(command);
        YT_ABORT(
            "Importing YT inline analysis script \"%s\" ... failed (please do not include the \".py\" extension)!\n",
            LibytProcessControl::Get().param_libyt_.script);
    }

    free(command);

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    // add imported script's namespace under in libyt.interactive_mode["script_globals"]
    command_width = 200 + strlen(LibytProcessControl::Get().param_libyt_.script);
    command = (char*)malloc(command_width * sizeof(char));
    sprintf(command,
            "libyt.interactive_mode[\"script_globals\"] = sys.modules[\"%s\"].__dict__\n"
            "libyt.interactive_mode[\"func_err_msg\"] = dict()\n"
            "libyt.interactive_mode[\"func_body\"] = dict()\n",
            LibytProcessControl::Get().param_libyt_.script);

    if (PyRun_SimpleString(command) == 0) {
        log_debug("Preparing interactive mode environment ... done\n");
    } else {
        free(command);
        YT_ABORT("Preparing interactive mode environment ... failed\n");
    }
    free(command);

    std::string filename = std::string(LibytProcessControl::Get().param_libyt_.script) + ".py";
    LibytPythonShell::load_file_func_body(filename.c_str());
    std::vector<std::string> func_list = LibytPythonShell::get_funcname_defined(filename.c_str());
    for (int i = 0; i < (int)func_list.size(); i++) {
        LibytProcessControl::Get().function_info_list_.AddNewFunction(func_list[i],
                                                                      FunctionInfo::RunStatus::kNotSetYet);
    }
#endif

    return YT_SUCCESS;

}  // FUNCTION : init_libyt_module
