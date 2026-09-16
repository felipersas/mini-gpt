"""Vocabulário: mapeamento bidirecional entre tokens e IDs inteiros."""

PAD = "<PAD>"  # preenche sequências até um tamanho comum
UNK = "<UNK>"  # token fora do vocabulário
BOS = "<BOS>"  # início de sequência
EOS = "<EOS>"  # fim de sequência

# Especiais sempre nos IDs 0..3; PAD = 0 faz um tensor de zeros ser padding.
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


class Vocab:
    """Tabela token <-> ID. O ID de um token é a sua posição em `id_to_token`."""

    pad_id = SPECIAL_TOKENS.index(PAD)
    unk_id = SPECIAL_TOKENS.index(UNK)
    bos_id = SPECIAL_TOKENS.index(BOS)
    eos_id = SPECIAL_TOKENS.index(EOS)

    def __init__(self, tokens: list[str]) -> None:
        # Remove duplicatas preservando a ordem; os especiais continuam em 0..3.
        self.id_to_token: list[str] = list(dict.fromkeys(SPECIAL_TOKENS + tokens))
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.id_to_token)}

    def __len__(self) -> int:
        return len(self.id_to_token)

    def get_id(self, token: str) -> int:
        """ID do token, ou `unk_id` se ele não estiver no vocabulário."""
        return self.token_to_id.get(token, self.unk_id)

    def get_token(self, token_id: int) -> str:
        """Token correspondente ao ID."""
        # Evita que IDs negativos indexem a lista pelo fim.
        if not 0 <= token_id < len(self):
            raise IndexError(f"ID {token_id} fora do vocabulário (0..{len(self) - 1})")
        return self.id_to_token[token_id]
