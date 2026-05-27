"""Tokenization utilities for SR2.

Public API:
  count_tokens(text: str) -> int
  truncate_to_tokens(text: str, budget: int) -> str
  TiktokenTokenCounter — TokenCounter protocol implementation (tiktoken-backed)
"""

from sr2.tokenization.counting import TiktokenTokenCounter, count_tokens, truncate_to_tokens

__all__ = ["TiktokenTokenCounter", "count_tokens", "truncate_to_tokens"]
