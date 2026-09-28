"""Deterministic MMR over a fixed candidate set and non-behavioral metadata."""
import numpy as np
import pandas as pd


class DiversityMetadata:
    def __init__(self, videos):
        if videos.video_id.duplicated().any():
            raise ValueError('Duplicate video metadata')
        self.videos = videos[['video_id', 'author_id', 'tag']].copy()
        self.lookup = {v: i for i, v in enumerate(videos.video_id)}
        self.authors = videos.author_id.fillna(-1).to_numpy()
        self.tags = [self.parse_tags(t) for t in videos.tag]
        vocabulary = sorted(set().union(*self.tags))
        columns = {t: i for i, t in enumerate(vocabulary)}
        self.features = np.zeros((len(videos), len(columns)), dtype=np.float32)
        for row, tags in enumerate(self.tags):
            for tag in tags:
                self.features[row, columns[tag]] = 1

    @staticmethod
    def parse_tags(value):
        if pd.isna(value):
            return set()
        return {t.strip() for t in str(value).split(',')
                if t.strip() not in ('', '-1', 'nan')}

    def similarities(self, items):
        indices = np.asarray([self.lookup[v] for v in items], dtype=int)
        features = self.features[indices]
        intersection = features @ features.T
        counts = features.sum(axis=1)
        union = counts[:, None] + counts[None, :] - intersection
        tags = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        authors = self.authors[indices]
        same_author = (authors[:, None] == authors[None, :]) & (authors[:, None] >= 0)
        # Missing metadata provides no positive evidence of similarity.
        return .5 * tags + .5 * same_author, tags, authors, counts > 0


def mmr_order(similarity, strength, k):
    """Greedy top-k, using base-rank percentile as relevance; ties favor base rank."""
    n = len(similarity)
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError('MMR strength must lie in [0, 1)')
    if not isinstance(k, int) or isinstance(k, bool) or not 0 < k <= n:
        raise ValueError('k must be within the candidate count')
    if similarity.shape != (n, n) or not np.isfinite(similarity).all():
        raise ValueError('Invalid similarity matrix')
    relevance = 1 - np.arange(n) / max(n - 1, 1)
    redundancy = np.zeros(n)
    available = np.ones(n, dtype=bool)
    chosen, values = [], []
    for _ in range(k):
        objective = (1 - strength) * relevance - strength * redundancy
        objective[~available] = -np.inf
        index = int(np.argmax(objective))
        chosen.append(index)
        values.append(float(objective[index]))
        available[index] = False
        redundancy = np.maximum(redundancy, similarity[:, index])
    return chosen, values


def rerank_candidates(frame, metadata, strength, k=20):
    """Select an ordered subset; never add candidates or consult outcome labels."""
    if frame.empty or frame.duplicated(['user_id', 'video_id']).any():
        raise ValueError('Candidates must be nonempty and unique per user')
    parts = []
    for _, group in frame.groupby('user_id', sort=True):
        group = group.sort_values(['final_rank', 'video_id'], kind='stable')
        if group.final_rank.tolist() != list(range(1, len(group) + 1)):
            raise ValueError('Candidate ranks must be complete and one-based')
        similarity, _, _, _ = metadata.similarities(group.video_id.tolist())
        order, values = mmr_order(similarity, strength, k)
        selected = group.iloc[order].copy()
        selected['base_rank'] = selected.final_rank
        selected['final_rank'] = np.arange(1, k + 1)
        selected['mmr_marginal_score'] = values
        parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def diversity_metrics(recommendations, metadata):
    """Macro metrics; tag diversity excludes pairs with missing tag metadata."""
    author_diversity, tag_diversity, max_author, tag_known, author_known = [], [], [], [], []
    for items in recommendations.values():
        _, tags, authors, known = metadata.similarities(items)
        known_authors = authors[authors >= 0]
        author_known.append(len(known_authors) / len(items))
        if len(known_authors):
            _, counts = np.unique(known_authors, return_counts=True)
            author_diversity.append(len(counts) / len(known_authors))
            max_author.append(float(counts.max() / len(known_authors)))
        pairs = np.triu(known[:, None] & known[None, :], k=1)
        if pairs.any():
            tag_diversity.append(float((1 - tags[pairs]).mean()))
        tag_known.append(float(known.mean()))
    return {
        'tag_ild': float(np.mean(tag_diversity)) if tag_diversity else None,
        'unique_author_fraction': float(np.mean(author_diversity)) if author_diversity else None,
        'max_author_share': float(np.mean(max_author)) if max_author else None,
        'tag_metadata_fraction': float(np.mean(tag_known)),
        'author_metadata_fraction': float(np.mean(author_known)),
        'tag_evaluated_users': len(tag_diversity),
    }


def select_strength(rows, max_relative_ndcg_loss):
    """Maximize tag diversity subject to a validation relevance floor."""
    if not 0 <= max_relative_ndcg_loss < 1:
        raise ValueError('Invalid relevance tolerance')
    baseline = next(r for r in rows if r['strength'] == 0)
    floor = baseline['metrics']['ndcg@20'] * (1 - max_relative_ndcg_loss)
    eligible = [r for r in rows if r['metrics']['ndcg@20'] >= floor]
    return max(eligible, key=lambda r: (
        r['metrics']['tag_ild'] if r['metrics']['tag_ild'] is not None else -1,
        r['metrics']['ndcg@20'], -r['strength']))
