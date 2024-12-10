from dataclasses import dataclass


@dataclass
class Condition:
    txt: str = None
    img: str = None
    single_strengths: list[float] = None
    double_strengths: list[float] = None
    start_index: int = None
    end_index: int = None


def inputs_to_conditions(
    prompt,
    prompt_2,
    redux,
    redux_2,
    prompt_single_strengths,
    prompt_double_strengths,
    prompt_2_single_strengths,
    prompt_2_double_strengths,
    redux_single_strengths,
    redux_double_strengths,
    redux_2_single_strengths,
    redux_2_double_strengths,
):
    if isinstance(prompt_single_strengths, str):
        prompt_single_strengths = [float(x) for x in prompt_single_strengths.split(",")]
    if isinstance(prompt_double_strengths, str):
        prompt_double_strengths = [float(x) for x in prompt_double_strengths.split(",")]
    if isinstance(prompt_2_single_strengths, str):
        prompt_2_single_strengths = [float(x) for x in prompt_2_single_strengths.split(",")]
    if isinstance(prompt_2_double_strengths, str):
        prompt_2_double_strengths = [float(x) for x in prompt_2_double_strengths.split(",")]
    if isinstance(redux_single_strengths, str):
        redux_single_strengths = [float(x) for x in redux_single_strengths.split(",")]
    if isinstance(redux_double_strengths, str):
        redux_double_strengths = [float(x) for x in redux_double_strengths.split(",")]
    if isinstance(redux_2_single_strengths, str):
        redux_2_single_strengths = [float(x) for x in redux_2_single_strengths.split(",")]
    if isinstance(redux_2_double_strengths, str):
        redux_2_double_strengths = [float(x) for x in redux_2_double_strengths.split(",")]

    conditions = []
    if prompt is not None:
        assert len(prompt_single_strengths) == 38
        assert len(prompt_double_strengths) == 19
        conditions.append(
            Condition(
                txt=prompt,
                img=None,
                single_strengths=prompt_single_strengths,
                double_strengths=prompt_double_strengths,
            )
        )

    if prompt_2 is not None:
        assert len(prompt_2_single_strengths) == 38
        assert len(prompt_2_double_strengths) == 19
        conditions.append(
            Condition(
                txt=prompt_2,
                img=None,
                single_strengths=prompt_2_single_strengths,
                double_strengths=prompt_2_double_strengths,
            )
        )

    if redux is not None:
        assert len(redux_single_strengths) == 38
        assert len(redux_double_strengths) == 19
        conditions.append(
            Condition(
                txt=None,
                img=redux,
                single_strengths=redux_single_strengths,
                double_strengths=redux_double_strengths,
            )
        )

    if redux_2 is not None:
        assert len(redux_2_single_strengths) == 38
        assert len(redux_2_double_strengths) == 19
        conditions.append(
            Condition(
                txt=None,
                img=redux_2,
                single_strengths=redux_2_single_strengths,
                double_strengths=redux_2_double_strengths,
            )
        )
    return conditions
