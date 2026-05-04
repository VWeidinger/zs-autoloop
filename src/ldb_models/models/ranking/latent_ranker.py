import torch
import torch.nn as nn
import torch.nn.functional as F
from ldb_models.training.training.outputs import ModelOutput


class LatentRankerBase(nn.Module):
    """
    Base class for latent rankers that score plans based on their latent representations.
    """

    def __init__(
        self,
        group_size: int,
        representation_dim: int,
        num_heads: int,
        num_layers: int,
        use_layernorm: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.group_size = group_size
        self.representation_dim = representation_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=representation_dim,
            nhead=num_heads,
            dim_feedforward=representation_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_layer = nn.Linear(representation_dim, 1)

        # LayerNorm can be optionally disabled via config
        self.use_layernorm = use_layernorm
        self.norm_layer = (
            nn.LayerNorm(representation_dim) if self.use_layernorm else None
        )

    def normalize(self, latent_representations: torch.Tensor) -> torch.Tensor:
        if self.use_layernorm and self.norm_layer is not None:
            return self.norm_layer(latent_representations)
        return latent_representations

    def pad_mask_group(self, latent_representations: torch.Tensor, num_items: int):
        """
        Pads `latent_representations` so its length is a multiple of `group_size`,
        creates the `key_padding_mask` and returns the reshaped groups.

        Returns: (groups, key_padding_mask, num_to_pad)
        """
        num_to_pad = (self.group_size - (num_items % self.group_size)) % self.group_size
        if num_to_pad > 0:
            padding = torch.zeros(
                num_to_pad,
                self.representation_dim,
                device=self.device,
                dtype=latent_representations.dtype,
            )
            latent_representations = torch.cat([latent_representations, padding], dim=0)
        num_groups = latent_representations.shape[0] // self.group_size
        key_padding_mask = torch.zeros(
            (num_groups, self.group_size), dtype=torch.bool, device=self.device
        )
        if num_to_pad > 0:
            key_padding_mask[-1, -num_to_pad:] = True
        groups = latent_representations.view(
            num_groups, self.group_size, self.representation_dim
        )
        return groups, key_padding_mask

    def apply_attention(
        self, groups: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_output = self.transformer_encoder(
            groups, src_key_padding_mask=key_padding_mask
        )
        scores = self.output_layer(attention_output)
        return scores

    def fill_padding_scores(
        self, scores: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensures padded tokens are never chosen when using min-winner logic
        by setting their scores to +inf.
        """
        if key_padding_mask is None:
            return scores
        # Ensure mask has same number of dims as scores for safe broadcasting
        if key_padding_mask.dim() == scores.dim() - 1:
            mask = key_padding_mask.unsqueeze(-1)
        else:
            mask = key_padding_mask
        scores = scores.masked_fill(mask, float("inf"))
        return scores

    def serialize_scores(
        self, scores_grouped: torch.Tensor, num_valid_items: int
    ) -> torch.Tensor:
        """
        Flattens grouped scores and truncates to `num_valid_items`.
        """
        scores_flat = scores_grouped.view(-1)[:num_valid_items]
        return scores_flat


class AttentionRanker(LatentRankerBase):
    """
    Applies self-attention to a fixed-size group of plan representations
    to produce a final score for each plan.
    """

    def forward(
        self, latent_representations: torch.Tensor, debug: bool = False, **kwargs
    ) -> torch.Tensor:
        num_items = latent_representations.shape[0]

        normalized_latents = self.normalize(latent_representations)
        groups, key_padding_mask = self.pad_mask_group(normalized_latents, num_items)
        scores = self.apply_attention(groups, key_padding_mask)
        scores_flat = self.serialize_scores(scores, num_items)
        final_scores = F.softplus(scores_flat)
        return ModelOutput(plan_predictions=final_scores)


class UnifiedAttentionRanker(LatentRankerBase):
    """
    Applies self-attention to a variable-length set of plan representations for a single query
    to produce a final score for each plan. This model is "unified" because it uses the
    same logic for both training and inference, relying on the DataLoader to provide
    batches where all plans belong to the same query.
    """

    def forward(self, latent_representations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Processes a set of plans for a single query.

        Args:
            latent_representations (torch.Tensor): A tensor of shape [num_plans, representation_dim]
                                                   containing the embeddings for all plans in the current batch
                                                   (which corresponds to a single query context).
            debug (bool): If True, prints shape information.

        Returns:
            torch.Tensor: A 1D tensor of shape [num_plans] containing the predicted score for each plan.
        """
        normalized_latents = self.normalize(latent_representations)
        num_items = latent_representations.shape[0]
        # No padding mask is needed since all plans are valid.
        transformer_input = normalized_latents.unsqueeze(0)
        scores = self.apply_attention(transformer_input, None)
        scores_flat = self.serialize_scores(scores, num_items)
        final_scores = F.softplus(scores_flat)
        assert final_scores.shape[0] == num_items, "Output scores length mismatch."
        return ModelOutput(plan_predictions=final_scores)


class TournamentAttentionRanker(LatentRankerBase):
    """
    Applies self-attention to a sequence of latent plan representations
    to produce a final score for each plan.

    Can operate in two modes:
    1. 'simple': Ranks a single batch of plans, padding them into fixed-size groups. (For training)
    2. 'tournament': Ranks a large set of plans for a query via an iterative
       elimination tournament. (For validation/testing)
    """

    def _predict_scores_for_groups(
        self, latent_representations: torch.Tensor, original_num_plans: int
    ) -> torch.Tensor:
        groups, key_padding_mask = self.pad_mask_group(
            latent_representations, original_num_plans
        )
        scores = self.apply_attention(groups, key_padding_mask)
        scores = self.fill_padding_scores(scores, key_padding_mask)
        return scores

    def forward(
        self,
        latent_representations: torch.Tensor,
        tournament_selection: bool = False,
        debug: bool = False,
    ) -> ModelOutput:
        original_num_plans = latent_representations.shape[0]
        latent_representations = self.normalize(latent_representations)
        # Decide on prediction mode
        if tournament_selection and original_num_plans > self.group_size:
            scores = self.tournament_predict(
                latent_representations, original_num_plans, debug
            )
        else:
            scores_grouped = self._predict_scores_for_groups(
                latent_representations, original_num_plans
            )
            scores = self.serialize_scores(scores_grouped, original_num_plans)
        # Apply non-linearity
        scores = F.softplus(scores)
        return ModelOutput(plan_predictions=scores)

    def winners_local_indices_from_scores_grouped(
        self, scores_grouped: torch.Tensor
    ) -> torch.Tensor:
        """Return the local (per-group) winner indices (argmin across group dimension)."""
        # scores_grouped has shape [num_groups, group_size, 1]; remove last dim first
        if scores_grouped.dim() == 3 and scores_grouped.size(-1) == 1:
            scores_2d = scores_grouped.squeeze(-1)
        else:
            scores_2d = scores_grouped
        return torch.argmin(scores_2d, dim=1)

    def winner_indices_from_scores_grouped(
        self, scores_grouped: torch.Tensor
    ) -> torch.Tensor:
        """Return the indices of the winners within a flat candidate list for the given grouped scores."""
        num_groups = scores_grouped.shape[0]
        group_offsets = (
            torch.arange(num_groups, device=self.device, dtype=torch.long)
            * self.group_size
        )
        winner_local_indices = self.winners_local_indices_from_scores_grouped(
            scores_grouped
        )
        # winner_local_indices is [num_groups], group_offsets is [num_groups]
        return group_offsets + winner_local_indices

    def update_candidate_indices(
        self,
        scores_flat: torch.Tensor,
        new_candidate_indices: torch.Tensor,
        num_groups: int,
        num_candidates_prev: int,
    ) -> torch.Tensor:
        # To select additional non-winners to carry over, find the non-winner indices
        is_winner_mask = torch.zeros(
            num_candidates_prev, dtype=torch.bool, device=self.device
        )
        is_winner_mask[new_candidate_indices] = True
        non_winner_indices_in_candidate_list = torch.where(~is_winner_mask)[0]
        # Get their scores
        non_winner_scores = scores_flat[non_winner_indices_in_candidate_list]
        # Determine how many non-winners to carry over
        num_needed_to_fill = max(0, self.group_size - num_groups)
        # Ensure we don't try to carry more non-winners than exist
        desired_num_to_carry = min(num_needed_to_fill, len(non_winner_scores))
        # Ensure we don't exceed total candidates - 1
        max_allowed = max(0, num_candidates_prev - num_groups - 1)
        # Final number of non-winners to carry
        num_non_winners_to_carry = min(desired_num_to_carry, max_allowed)
        if num_non_winners_to_carry > 0:
            _, top_non_winner_indices = torch.topk(
                non_winner_scores, num_non_winners_to_carry, largest=False
            )
            new_candidate_indices = torch.cat(
                [
                    new_candidate_indices,
                    non_winner_indices_in_candidate_list[top_non_winner_indices],
                ]
            )
        return new_candidate_indices

    def select_candidates_for_next_round(
        self,
        candidate_indices: torch.Tensor,
        scores_grouped: torch.Tensor,
        scores_flat: torch.Tensor,
        num_candidates: int,
    ) -> None:
        num_groups = scores_grouped.shape[0]
        selected_candidate_indices = self.winner_indices_from_scores_grouped(
            scores_grouped
        )
        num_candidates_prev = num_candidates
        # Handle underfilled last group by carrying non-winners if needed
        if (num_groups < self.group_size) and (num_candidates_prev > self.group_size):
            selected_candidate_indices = self.update_candidate_indices(
                scores_flat, selected_candidate_indices, num_groups, num_candidates_prev
            )
        # Now get the winners' original indices to form the candidate list for the next round
        new_candidate_indices = candidate_indices[selected_candidate_indices]
        # Update the number of candidates and return the new list
        num_candidates = new_candidate_indices.shape[0]
        return new_candidate_indices, num_candidates

    def update_predictions(
        self,
        all_predictions: torch.Tensor,
        candidate_indices: torch.Tensor,
        scores_flat: torch.Tensor,
    ) -> None:
        """Updates `all_predictions` with the latest round's scores."""
        all_predictions += scores_flat.max()
        all_predictions[candidate_indices] = scores_flat

    def tournament_predict(
        self,
        latent_representations: torch.Tensor,
        original_num_plans: int,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Performs iterative elimination tournament to rank a large set of plans.
        """
        round_num = 1
        num_candidates = original_num_plans
        all_predictions = torch.zeros(original_num_plans, device=self.device)
        candidate_indices = torch.arange(original_num_plans, device=self.device)
        if debug:
            print(f"Starting tournament with {original_num_plans} candidates.")
        while num_candidates > 1:
            candidate_latents = latent_representations[candidate_indices]
            # Predict scores for current candidates
            scores_grouped = self._predict_scores_for_groups(
                candidate_latents, num_candidates
            )
            # Flatten the scores for easier indexing and remove padding scores
            scores_flat = self.serialize_scores(scores_grouped, num_candidates)
            # Update predictions with current round's results
            self.update_predictions(all_predictions, candidate_indices, scores_flat)
            # Determine winners for this round (lower score wins)
            candidate_indices, num_candidates = self.select_candidates_for_next_round(
                candidate_indices, scores_grouped, scores_flat, num_candidates
            )
            if debug:
                print(
                    f"Tournament round {round_num} complete. Remaining candidates: {num_candidates}"
                )
            round_num += 1
        return all_predictions
