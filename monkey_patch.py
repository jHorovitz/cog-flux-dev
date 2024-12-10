import torch

from diffusers import FluxPipeline


def monkey_patch_pipeline(pipe: FluxPipeline):
    """
    Wraps transformer blocks and passes the block_index in through the joint_attention_kwargs.
    """
    pipe.transformer.transformer_blocks = torch.nn.ModuleList(
        [
            TransformerBlockWrapper.maybe_wrap(block_index, block)
            for block_index, block in enumerate(pipe.transformer.transformer_blocks)
        ]
    )
    pipe.transformer.single_transformer_blocks = torch.nn.ModuleList(
        [
            TransformerBlockWrapper.maybe_wrap(block_index, block)
            for block_index, block in enumerate(pipe.transformer.single_transformer_blocks)
        ]
    )


class TransformerBlockWrapper(torch.nn.Module):
    """This class is a wrapper around the transformer blocks in the model to add block_index and block_class to the joint_attention_kwargs."""

    def __init__(self, block_index, block):
        super(TransformerBlockWrapper, self).__init__()
        self.block_index = block_index
        self.block = block

    def forward(self, *args, joint_attention_kwargs=None, **kwargs):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs["block_index"] = self.block_index
            joint_attention_kwargs["block_class"] = self.block.__class__.__name__

        return self.block(*args, joint_attention_kwargs=joint_attention_kwargs, **kwargs)

    @staticmethod
    def maybe_wrap(block_index, block):
        if isinstance(block, TransformerBlockWrapper):
            # don't rewrap if it's already wrapped.
            return block
        else:
            return TransformerBlockWrapper(block_index, block)
