#include <gtest/gtest.h>
#ifndef SERIAL_MODE
#include "comm_mpi.h"
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
        CommMpi::InitializeInfo(0);
#endif
        LibytPythonShell::SetMPIInfo(mpi_size_, 0, mpi_rank_);

        // Create namespace for PythonShell to execute code
        InitializeAndImportScript(script_);
        LibytPythonShell::SetExecutionNamespace(GetScriptPyNamespace(script_));
        LibytPythonShell::SetFunctionBodyDict(CreateTemplateDictStorage());
    }

    void TearDown() override { PyRun_SimpleString("del sys"); }

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
    PyObject* CreateTemplateDictStorage() {
        PyRun_SimpleString("import sys; sys.TEMPLATE_DICT_STORAGE = dict()");
        PyObject* py_sys = PyImport_ImportModule("sys");
        PyObject* py_template_dict_storage = PyObject_GetAttrString(py_sys, "TEMPLATE_DICT_STORAGE");
        Py_DECREF(py_sys);
        Py_DECREF(py_template_dict_storage);
        return py_template_dict_storage;
    }
};

class TestPythonShell : public PythonFixture {};

TEST_F(TestPythonShell, AllExecutePrompt_can_execute_rvalue_empty_code) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    int output_mpi_rank = 0;
    std::string expected_output;

    // Act
    std::vector<PythonOutput> output;
    PythonStatus status;
    if (GetMpiRank() == src_mpi_rank) {
        status = python_shell_.AllExecutePrompt("", "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecutePrompt("", "", src_mpi_rank, output, output_mpi_rank);
    }

    // Assert
    EXPECT_EQ(status, PythonStatus::kPythonSuccess);
    EXPECT_EQ(GetMpiSize(), output.size()) << "Output size is not equal to MPI size.";
    if (GetMpiRank() == output_mpi_rank) {
        for (int r = 0; r < output.size(); r++) {
            EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
            EXPECT_EQ(output[r].output, expected_output);
            EXPECT_EQ(output[r].error, "");
        }
    } else {
        for (int r = 0; r < output.size(); r++) {
            if (r == GetMpiRank()) {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
            } else {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonUnknown);
            }
            EXPECT_EQ(output[r].output, expected_output);
            EXPECT_EQ(output[r].error, "");
        }
    }
}

TEST_F(TestPythonShell, AllExecutePrompt_can_execute_a_valid_single_statement) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    int output_mpi_rank = 0;
    std::string src_code = "print('hello')";  // TODO: parameterize this
    std::string expected_output = "hello\n";

    // Act
    std::vector<PythonOutput> output;
    PythonStatus status;
    if (GetMpiRank() == src_mpi_rank) {
        status = python_shell_.AllExecutePrompt(src_code, "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecutePrompt("", "", src_mpi_rank, output, output_mpi_rank);
    }

    // Assert
    EXPECT_EQ(status, PythonStatus::kPythonSuccess);
    EXPECT_EQ(GetMpiSize(), output.size()) << "Output size is not equal to MPI size.";
    if (GetMpiRank() == output_mpi_rank) {
        for (int r = 0; r < output.size(); r++) {
            EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
            EXPECT_EQ(output[r].output, expected_output);
            EXPECT_EQ(output[r].error, "");
        }
    } else {
        for (int r = 0; r < output.size(); r++) {
            if (r == GetMpiRank()) {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
                EXPECT_EQ(output[r].output, expected_output);
            } else {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonUnknown);
                EXPECT_EQ(output[r].output, "");
            }
            EXPECT_EQ(output[GetMpiRank()].error, "");
        }
    }
}

TEST_F(TestPythonShell, AllExecuteFile_can_execute_a_valid_arbitrary_code) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    int output_mpi_rank = 0;
    // TODO: probably should write a correct answer generator for the test
    std::string src_code = "\n"
                           "from mpi4py import MPI\n"
                           ""
                           "def func(mpi_rank):\n"
                           "    print(mpi_rank)\n"
                           "\n"
                           "func(MPI.COMM_WORLD.Get_rank())\n"
                           "\n"
                           "print('hello')\n";  // TODO: parameterize this
    std::vector<std::string> expected_output;
    expected_output.reserve(GetMpiSize());
    for (int r = 0; r < GetMpiSize(); r++) {
        expected_output.emplace_back(std::to_string(r) + "\nhello\n");
    }

    // Act
    std::vector<PythonOutput> output;
    PythonStatus status;
    if (GetMpiRank() == src_mpi_rank) {
        status = python_shell_.AllExecuteFile(src_code, "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecuteFile("", "", src_mpi_rank, output, output_mpi_rank);
    }

    // Assert
    EXPECT_EQ(status, PythonStatus::kPythonSuccess);
    EXPECT_EQ(GetMpiSize(), output.size()) << "Output size is not equal to MPI size.";
    if (GetMpiRank() == output_mpi_rank) {
        for (int r = 0; r < output.size(); r++) {
            EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
            EXPECT_EQ(output[r].output, expected_output[r]);
            EXPECT_EQ(output[r].error, "");
        }
    } else {
        for (int r = 0; r < output.size(); r++) {
            if (r == GetMpiRank()) {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
                EXPECT_EQ(output[r].output, expected_output[r]);
            } else {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonUnknown);
                EXPECT_EQ(output[r].output, "");
            }
            EXPECT_EQ(output[GetMpiRank()].error, "");
        }
    }
}

TEST_F(TestPythonShell, AllExecuteCell_can_execute_empty_code) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    int output_mpi_rank = 0;
    // TODO: probably should write a correct answer generator for the test
    std::string src_code = "";  // TODO: parameterize this
    std::vector<std::string> expected_output(GetMpiSize(), "");

    // Act
    std::vector<PythonOutput> output;
    PythonStatus status;
    if (GetMpiRank() == src_mpi_rank) {
        status = python_shell_.AllExecuteCell(src_code, "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecuteCell("", "", src_mpi_rank, output, output_mpi_rank);
    }

    // Assert
    EXPECT_EQ(status, PythonStatus::kPythonSuccess);
    EXPECT_EQ(GetMpiSize(), output.size()) << "Output size is not equal to MPI size.";
    if (GetMpiRank() == output_mpi_rank) {
        for (int r = 0; r < output.size(); r++) {
            EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
            EXPECT_EQ(output[r].output, expected_output[r]);
            EXPECT_EQ(output[r].error, "");
        }
    } else {
        for (int r = 0; r < output.size(); r++) {
            if (r == GetMpiRank()) {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
                EXPECT_EQ(output[r].output, expected_output[r]);
            } else {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonUnknown);
                EXPECT_EQ(output[r].output, "");
            }
            EXPECT_EQ(output[GetMpiRank()].error, "");
        }
    }
}

TEST_F(TestPythonShell, AllExecuteCell_can_execute_a_valid_arbitrary_code) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    int output_mpi_rank = 0;
    // TODO: probably should write a correct answer generator for the test
    std::string src_code = "\n"
                           "from mpi4py import MPI\n"
                           ""
                           "def func(mpi_rank):\n"
                           "    print(mpi_rank)\n"
                           "\n"
                           "func(MPI.COMM_WORLD.Get_rank())\n"
                           "\n"
                           "print('hello')\n";  // TODO: parameterize this
    std::vector<std::string> expected_output;
    expected_output.reserve(GetMpiSize());
    for (int r = 0; r < GetMpiSize(); r++) {
        expected_output.emplace_back(std::to_string(r) + "\nhello\n");
    }

    // Act
    std::vector<PythonOutput> output;
    PythonStatus status;
    if (GetMpiRank() == src_mpi_rank) {
        status = python_shell_.AllExecuteCell(src_code, "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecuteCell("", "", src_mpi_rank, output, output_mpi_rank);
    }

    // Assert
    EXPECT_EQ(status, PythonStatus::kPythonSuccess);
    EXPECT_EQ(GetMpiSize(), output.size()) << "Output size is not equal to MPI size.";
    if (GetMpiRank() == output_mpi_rank) {
        for (int r = 0; r < output.size(); r++) {
            EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
            EXPECT_EQ(output[r].output, expected_output[r]);
            EXPECT_EQ(output[r].error, "");
        }
    } else {
        for (int r = 0; r < output.size(); r++) {
            if (r == GetMpiRank()) {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonSuccess);
                EXPECT_EQ(output[r].output, expected_output[r]);
            } else {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonUnknown);
                EXPECT_EQ(output[r].output, "");
            }
            EXPECT_EQ(output[GetMpiRank()].error, "");
        }
    }
}

TEST_F(TestPythonShell, AllExecuteCell_can_resolve_an_invalid_code) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    int output_mpi_rank = 0;
    // TODO: probably should write a correct answer generator for the test
    std::string src_code = "\n"
                           "from mpi4py import MPI\n"
                           ""
                           "def func(mpi_rank):\n"
                           "    print(mpi_rank\n"
                           "\n"
                           "func(MPI.COMM_WORLD.Get_rank())\n"
                           "\n"
                           "print('hello')\n";  // TODO: parameterize this
    std::vector<std::string> expected_output;
    expected_output.reserve(GetMpiSize());
    for (int r = 0; r < GetMpiSize(); r++) {
        expected_output.emplace_back(std::to_string(r) + "\nhello\n");
    }

    // Act
    std::vector<PythonOutput> output;
    PythonStatus status;
    if (GetMpiRank() == src_mpi_rank) {
        status = python_shell_.AllExecuteCell(src_code, "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecuteCell("", "", src_mpi_rank, output, output_mpi_rank);
    }

    // Assert
    EXPECT_EQ(status, PythonStatus::kPythonFailed);
    EXPECT_EQ(GetMpiSize(), output.size()) << "Output size is not equal to MPI size.";
    if (GetMpiRank() == output_mpi_rank) {
        for (int r = 0; r < output.size(); r++) {
            EXPECT_EQ(output[r].status, PythonStatus::kPythonFailed);
        }
    } else {
        for (int r = 0; r < output.size(); r++) {
            if (r == GetMpiRank()) {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonFailed);
            } else {
                EXPECT_EQ(output[r].status, PythonStatus::kPythonUnknown);
            }
        }
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