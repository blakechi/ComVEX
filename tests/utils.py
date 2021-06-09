import torch

def assert_output_has_nan(x: torch.Tensor) -> None:
    assert torch.isnan(x).any() == False, "Output contains NaN."

def assert_output_shape_wrong(x: torch.Tensor, expected_shape: tuple) -> None:
    assert (
        x.shape == expected_shape
    ), f"Output's shape: {tuple(x.shape)} != Expected shape: {expected_shape}."