#include <gtest/gtest.h>

TEST(MacroTest, Demonstrate) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(7 * 6, 42);
    EXPECT_TRUE(true);
    EXPECT_STREQ("shin", "shin") << "This is custom failure message!!!";
    EXPECT_FALSE(false);
}

// Using fixtures by defining a class and let other tests inherits it.
class Fixtures : public testing::Test {
private:
    int* x;

    void SetUp() override {
        std::cout << "Setting up ..." << std::endl;
        x = new int(100);
    }

    void TearDown() override {
        std::cout << "Tearing down ..." << std::endl;
        delete x;
    }

protected:
    int GetX() { return *x; }
    void SetX(int value) { *x = value; }
};

class FixtureTests : public Fixtures {};

TEST_F(FixtureTests, BasicAssertions) {
    SetX(10);
    EXPECT_EQ(10, GetX());

    SetX(1);
    EXPECT_EQ(1, GetX()) << "X is " << GetX();
}

// Using parameterized fixture with single parameter
class SinglePara : public testing::TestWithParam<int> {
protected:
    void Print(int value) { std::cout << "value = " << value << std::endl; }
};
TEST_P(SinglePara, Demonstration) {
    int value = GetParam();
    Print(value);
}

INSTANTIATE_TEST_SUITE_P(SingleParaTestSuite, SinglePara, testing::Values(1, 2, 100));

// Using parameterized fixture with multi parameter
class MultiPara : public testing::TestWithParam<std::tuple<int, bool>> {
protected:
    void Print(std::tuple<int, bool>& values) {
        std::cout << "(" << std::get<0>(values) << ", " << (std::get<1>(values) ? "true" : "false") << ") "
                  << std::endl;
    }
};
TEST_P(MultiPara, Demonstration) {
    int value0 = std::get<0>(GetParam());
    bool value1 = std::get<1>(GetParam());
    std::tuple<int, bool> value = std::make_tuple(value0, value1);
    Print(value);
}

INSTANTIATE_TEST_SUITE_P(MultiParaTestSuite, MultiPara,
                         testing::Values(std::make_tuple(100, false), std::make_tuple(110, true)));

// Mixing fixtures and parameterizing
// Because TestWithParam<T> inherits both from Test and WithParamInterface<T>,
// we only need to inherit WithParamInterface.
class MixingParameterizedAndFixtures : public FixtureTests, public testing::WithParamInterface<int> {};
TEST_P(MixingParameterizedAndFixtures, Demonstration) {
    int value = GetParam();
    std::cout << "[Para] value = " << value << std::endl;
    std::cout << "[Fixture] X = " << GetX() << std::endl;
}

INSTANTIATE_TEST_SUITE_P(MixingParameterizedAndFixturesTestSuite, MixingParameterizedAndFixtures,
                         testing::Values(1, 2, 3));
