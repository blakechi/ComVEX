# ComVEX: Computer Vision EXpo

Hi there! This is a reimplementation library for computer vision models by **PyTorch** and **Einops**. Our mission is to bridge papers and codes with consistent and clear implementations.

## What are the pros?

1. Consistent Structure \
   Every models share similar building objects:

   - `xxxBase`: A model's base. For checking common input arguments and storing important variables. Sometimes it can also provide specified weight initialization methods or necessary tensor operations, like patching and flattening images in ViT.
   - `xxxBackbone`: A model's backbone architecture. It includes every needed components to build the model except the classifier.
   - `xxxWithLinearClassifier`: `xxxBackbone` plus a projection head as its classifier. Only accept `xxxConfig` as its argument. Similar to Huggingface. Might provide some variants for differenet objective in the future.
   - `xxxConfig`: A configuration for all possible coefficients. It also provides model specializations mentioned in the papers.

2. Consistent Namings for papers and across papers \
   To make researchers or developers understand implementations as soon as possible, we tightly follow the names of model components from the official papers and be consistent on common namings appeared across papers.

3. Clear Tensor Operations \
   We use **Einops** for almost all tensor operations to unveil the dimensions of tensors, which are usually hidden in the code, and make our implementations explain by themselves.

4. Clear Arguments \
   To expose all possible arguments to users but still remain convenience, we categorize building objects into a hierarchical order with 3 levels listed from bottom to top:

   - **Basic** : The paper-proposed and essential objects that inherit directly from `nn.Module`, like `ViTBase`, `MultiheadAttention`, `SpatialGatingUnit`, etc.
   - **Wrapper**: Intermediate objects that organize basic ones, like `TransformerEncoderLayer`, `PerceiverBlock`, `MLPMixerLayer`, `xxxWithLinearClassifier`, etc.
   - **Model**: `xxxBackbone` and `xxxConfig`.

   **Basic** and **Model** objects are the ones crucial for paper-to-code mappings and model usages, so we require their arguments to be fully explicit in `__init__` methods. And for the sake of convenience, **Wrapper** objects can use `args` or `**kwargs` to pass down necessary arguments. The overall model structures in term of the number of required arguments will look like a hourglass.

5. Semantic Naming \
   Excluding some common names like `x` for the input tensors, `ff_dropout` for the dropout rate of feed forward networks, and `act_func_name` for a string of activation function's name, all variables, helper functions, and objects should be named meaningfully.

6. Detailed Model Information \
   Every models has its own `README.md` file that provides usages, one-by-one argument explanations, and usable objects. The official implementations are provided as well if any mentioned in the official paper.

## Notes

- Continuously implementing models, please check them out under the `comvex` folder for more details and `examples` folder for some demos.
- Pull requests are welcome!
