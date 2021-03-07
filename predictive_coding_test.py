from absl.testing import absltest
from absl.testing import parameterized

from predictive_coding import infer_pc, learn_pc


def learn_pc_test_data():
    return (
        dict(
            testcase_name="base_case",
            dummy=True,
        ),
    )


def infer_pc_test_data():
    return (
        dict(
            testcase_name="base_case",
            dummy=True,
        ),
    )


class PredCodingTest(parameterized.TestCase):
    @parameterized.named_parameters(*learn_pc_test_data())
    def test_learn_pc(
        self,
        dummy,
    ):
        pass

    @parameterized.named_parameters(*infer_pc_test_data())
    def test_infer_pc(
        self,
        dummy,
    ):
        pass


if __name__ == "__main__":
    absltest.main()
