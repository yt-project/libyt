#include <gtest/gtest.h>
#ifndef SERIAL_MODE
#include <mpi.h>
#endif
#include <fstream>

#include "libyt_python_shell.h"

class PythonFixture : public testing::Test {
private:
    // Though mpi_ prefix is used, in serial mode, it will be rank 0 and size 1.
    int mpi_rank_ = 0;
    int mpi_size_ = 1;

    // Neglect ".py" extension
    std::string script_ = "inline_script";

    void SetUp() override {
#ifndef SERIAL_MODE
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
#endif
        LibytPythonShell::SetMPIInfo(mpi_size_, 0, mpi_rank_);

        // Create namespace for PythonShell to execute code
        InitializeAndImportScript(script_);
        GetScriptPyNamespace(script_);
        LibytPythonShell::SetExecutionNamespace(GetScriptPyNamespace(script_));
    }

protected:
    LibytPythonShell python_shell_;
    int GetMpiRank() const { return mpi_rank_; }
    int GetMpiSize() const { return mpi_size_; }
    void InitializeAndImportScript(const std::string& script) {
        if (mpi_rank_ == 0) {
            // Create a script file
            struct stat buffer;
            std::string script_fullname = script + std::string(".py");
            if (stat(script_fullname.c_str(), &buffer) == 0) {
                std::cout << "Test script '" << script_fullname << "' ... found" << std::endl;
            } else {
                std::ofstream python_script(script_fullname.c_str());
                python_script.close();
                std::cout << "Empty test script '" << script_fullname << "' ... created" << std::endl;
            }
        }
#ifndef SERIAL_MODE
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // Import the script
        std::string statement = "import " + script;
        PyRun_SimpleString("import sys; sys.path.insert(0, '.')");
        PyRun_SimpleString(statement.c_str());
    }
    PyObject* GetScriptPyNamespace(const std::string& script) {
        PyObject* py_sys = PyImport_ImportModule("sys");
        PyObject* py_modules = PyObject_GetAttrString(py_sys, "modules");
        PyObject* py_script_module = PyDict_GetItemString(py_modules, script.c_str());
        PyObject* py_script_namespace = PyModule_GetDict(py_script_module);
        Py_DECREF(py_sys);
        Py_DECREF(py_modules);
        return py_script_namespace;
    }
};

class TestPythonShell : public PythonFixture {};

TEST_F(TestPythonShell, AllExecutePrompt_can_execute_rvalue_empty_code) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;

    // Act
    std::vector<PythonOutput> output;
    if (GetMpiRank() == src_mpi_rank) {
        python_shell_.AllExecutePrompt("", "<test>", src_mpi_rank, output);
    } else {
        python_shell_.AllExecutePrompt("", "", src_mpi_rank, output);
    }

    // Assert
    EXPECT_EQ(GetMpiSize(), output.size()) << "Output size is not equal to MPI size.";
    for (int r = 0; r < output.size(); r++) {
        EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
        EXPECT_EQ(output[r].output, "");
        EXPECT_EQ(output[r].error, "");
    }
}

TEST_F(TestPythonShell, AllExecutePrompt_can_execute_a_valid_single_statement) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    std::string src_code = "print('hello')";  // TODO: parameterize this

    // Act
    std::vector<PythonOutput> output;
    if (GetMpiRank() == src_mpi_rank) {
        python_shell_.AllExecutePrompt(src_code, "<test>", src_mpi_rank, output);
    } else {
        python_shell_.AllExecutePrompt("", "", src_mpi_rank, output);
    }

    // Assert
    EXPECT_EQ(GetMpiSize(), output.size()) << "Output size is not equal to MPI size.";
    for (int r = 0; r < output.size(); r++) {
        EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
        EXPECT_EQ(output[r].output, "hello");
        EXPECT_EQ(output[r].error, "");
    }
}

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
#ifndef SERIAL_MODE
    MPI_Init(&argc, &argv);
#endif
    // initialize python (todo: should call libyt api later)
    wchar_t* program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);
    Py_Initialize();

    result = RUN_ALL_TESTS();

    // finalize python (todo: should call libyt api later)
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);
#ifndef SERIAL_MODE
    MPI_Finalize();
#endif

    return result;
}