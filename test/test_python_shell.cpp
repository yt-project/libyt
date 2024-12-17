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

        // Initialize not-done-yet error msg
        LibytPythonShell::InitializeNotDoneErrMsg();

        // Create namespace for PythonShell to execute code
        InitializeAndImportScript(script_);
        std::cout << "LibytPythonShell will execute code in namespace '" << script_ << "'" << std::endl;
        LibytPythonShell::SetExecutionNamespace(GetScriptPyNamespace(script_));
        LibytPythonShell::SetFunctionBodyDict(CreateTemplateDictStorage());
    }

    void TearDown() override { PyRun_SimpleString("del sys"); }

protected:
    LibytPythonShell python_shell_;
    int GetMpiRank() const { return mpi_rank_; }
    int GetMpiSize() const { return mpi_size_; }
    void InitializeAndImportScript(const std::string& script) const {
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
    static PyObject* GetScriptPyNamespace(const std::string& script) {
        PyObject* py_sys = PyImport_ImportModule("sys");
        PyObject* py_modules = PyObject_GetAttrString(py_sys, "modules");
        PyObject* py_script_module = PyDict_GetItemString(py_modules, script.c_str());
        PyObject* py_script_namespace = PyModule_GetDict(py_script_module);
        Py_DECREF(py_sys);
        Py_DECREF(py_modules);
        return py_script_namespace;
    }
    static PyObject* CreateTemplateDictStorage() {
        PyRun_SimpleString("import sys; sys.TEMPLATE_DICT_STORAGE = dict()");
        PyObject* py_sys = PyImport_ImportModule("sys");
        PyObject* py_template_dict_storage = PyObject_GetAttrString(py_sys, "TEMPLATE_DICT_STORAGE");
        Py_DECREF(py_sys);
        Py_DECREF(py_template_dict_storage);
        return py_template_dict_storage;
    }
    static std::string GenerateFullErrMsg(const std::string& code) {
        PyObject *py_src, *py_exc, *py_val;
        std::string err_msg_str;

        py_src = Py_CompileString(code.c_str(), "<get err msg>", Py_single_input);

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12
        py_exc = PyErr_GetRaisedException();
        py_val = PyObject_GetAttrString(py_exc, "msg");
        err_msg_str = std::string(PyUnicode_AsUTF8(py_val));
#else
        PyObject *py_traceback, *py_obj;
        const char* err_msg;
        PyErr_Fetch(&py_exc, &py_val, &py_traceback);
        PyArg_ParseTuple(py_val, "sO", &err_msg, &py_obj);
        err_msg_str = std::string(err_msg);
#endif

        // dereference
        Py_XDECREF(py_src);
        Py_XDECREF(py_val);
        Py_XDECREF(py_exc);
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12
#else
        Py_XDECREF(py_traceback);
        Py_XDECREF(py_obj);
#endif
        PyErr_Clear();

        return err_msg_str;
    }
};

class TestPythonExecution : public PythonFixture {};
class TestCheckCodeValidity : public PythonFixture, public testing::WithParamInterface<std::string> {};
class TestCheckCodeValidityAssumption : public PythonFixture, public testing::WithParamInterface<std::string> {};

TEST_F(TestPythonExecution, AllExecutePrompt_can_execute_rvalue_empty_code) {
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

TEST_F(TestPythonExecution, AllExecutePrompt_can_execute_a_valid_single_statement) {
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

TEST_F(TestPythonExecution, AllExecutePrompt_can_resolve_an_arbitrary_code) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    int output_mpi_rank = 0;
    std::string src_code = "if True:\n"
                           "    print()\n"
                           "print('hello')\n";  // TODO: parameterize this

    // Act
    std::vector<PythonOutput> output;
    PythonStatus status;
    if (GetMpiRank() == src_mpi_rank) {
        status = python_shell_.AllExecutePrompt(src_code, "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecutePrompt("", "", src_mpi_rank, output, output_mpi_rank);
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

TEST_F(TestPythonExecution, AllExecutePrompt_can_resolve_an_invalid_single_statement) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;
    // Arrange
    int src_mpi_rank = 0;
    int output_mpi_rank = 0;
    std::string src_code = "print('hello'\n";  // TODO: parameterize this

    // Act
    std::vector<PythonOutput> output;
    PythonStatus status;
    if (GetMpiRank() == src_mpi_rank) {
        status = python_shell_.AllExecutePrompt(src_code, "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecutePrompt("", "", src_mpi_rank, output, output_mpi_rank);
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

TEST_F(TestPythonExecution, AllExecuteFile_can_execute_a_valid_arbitrary_code) {
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

TEST_F(TestPythonExecution, AllExecuteFile_can_resolve_an_invalid_arbitrary_code) {
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
        status = python_shell_.AllExecuteFile(src_code, "<test>", src_mpi_rank, output, output_mpi_rank);
    } else {
        status = python_shell_.AllExecuteFile("", "", src_mpi_rank, output, output_mpi_rank);
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

TEST_F(TestPythonExecution, AllExecuteCell_can_execute_empty_code) {
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

TEST_F(TestPythonExecution, AllExecuteCell_can_execute_a_valid_arbitrary_code) {
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

TEST_F(TestPythonExecution, AllExecuteCell_can_resolve_an_invalid_arbitrary_code) {
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

TEST_P(TestCheckCodeValidity, Can_distinguish_user_not_done_yet_error) {
    std::cout << "mpi_size = " << GetMpiSize() << ", mpi_rank = " << GetMpiRank() << std::endl;

    // Arrange
    std::string not_done_code = GetParam();

    // Act
    CodeValidity code_validity = LibytPythonShell::CheckCodeValidity(not_done_code, true, "<test>");

    // Assert
    EXPECT_EQ(code_validity.is_valid, "incomplete");
}

TEST_P(TestCheckCodeValidityAssumption, Error_lineno_should_be_at_the_end) {
    // Arrange
    std::string not_done_code = GetParam();

    // Act -- Get error message
    std::string full_err_msg = GenerateFullErrMsg(not_done_code);

    // Assert -- If there is number in it, it should be at the end of the error msg
    std::size_t found_number = full_err_msg.find_first_of("0123456789");
    if (found_number != std::string::npos) {
        EXPECT_LE(full_err_msg.length() - found_number, 3);
    }
}

INSTANTIATE_TEST_SUITE_P(IsIncompleteInPrompt, TestCheckCodeValidity,
                         testing::Values(std::string("if 1==1:\n"), std::string("if 1==1:\n  pass\nelse:\n"),
                                         std::string("if 1==1:\n  pass\nelif 2==2:\n"), std::string("try:\n"),
                                         std::string("try:\n  pass\nexcept:\n"),
                                         std::string("try:\n  pass\nfinally:\n"), std::string("class A:\n"),
                                         std::string("for _ in range(1):\n"), std::string("def func():\n"),
                                         std::string("while(False):\n"), std::string("with open('') as f:\n"),
                                         std::string("\"\"\"\n"), std::string("'''\n"), std::string("r\"\"\"\n"),
                                         std::string("u\"\"\"\n"), std::string("f\"\"\"\n"), std::string("b\"\"\"\n"),
                                         std::string("rf\"\"\"\n"), std::string("rb\"\"\"\n"), std::string("r'''\n"),
                                         std::string("u'''\n"), std::string("f'''\n"), std::string("b'''\n"),
                                         std::string("rf'''\n"), std::string("rb'''\n"), std::string("(\n"),
                                         std::string("[\n"), std::string("{\n")
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 10
                                                                 ,
                                         std::string("match (100):\n"), std::string("match (100):\n  case 100:\n")
#endif
                                             ));

INSTANTIATE_TEST_SUITE_P(ErrorMsgHasNumberAtTheEnd, TestCheckCodeValidityAssumption,
                         testing::Values(std::string("if 1==1:\n"), std::string("if 1==1:\n  pass\nelse:\n"),
                                         std::string("if 1==1:\n  pass\nelif 2==2:\n"), std::string("try:\n"),
                                         std::string("try:\n  pass\nexcept:\n"),
                                         std::string("try:\n  pass\nfinally:\n"), std::string("class A:\n"),
                                         std::string("for _ in range(1):\n"), std::string("def func():\n"),
                                         std::string("while(False):\n"), std::string("with open('') as f:\n"),
                                         std::string("\"\"\"\n"), std::string("'''\n"), std::string("r\"\"\"\n"),
                                         std::string("u\"\"\"\n"), std::string("f\"\"\"\n"), std::string("b\"\"\"\n"),
                                         std::string("rf\"\"\"\n"), std::string("rb\"\"\"\n"), std::string("r'''\n"),
                                         std::string("u'''\n"), std::string("f'''\n"), std::string("b'''\n"),
                                         std::string("rf'''\n"), std::string("rb'''\n"), std::string("(\n"),
                                         std::string("[\n"), std::string("{\n")
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 10
                                                                 ,
                                         std::string("match (100):\n"), std::string("match (100):\n  case 100:\n")
#endif
                                             ));

int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
#ifndef SERIAL_MODE
    MPI_Init(&argc, &argv);
#endif
    // initialize python
    wchar_t* program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);
    Py_Initialize();

    result = RUN_ALL_TESTS();

    // finalize python
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);
#ifndef SERIAL_MODE
    MPI_Finalize();
#endif

    return result;
}