"""
Compatibility shim for docs/spec naming.

This package follows encoding-model practice discussed in the LITcoder literature
(GT-LIT-Lab), but **does not** vendor-wrap the upstream `litcoder` repository.
Use [`alignment.encoding_pipeline.EncodingAlignmentPipeline`](alignment.encoding_pipeline)
for the implementation.
"""

from alignment.encoding_pipeline import EncodingAlignmentPipeline as LITcoderAlignmentPipeline

__all__ = ["LITcoderAlignmentPipeline"]
