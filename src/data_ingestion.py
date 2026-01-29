"""Data Ingestion Module with Demo Papers"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class Paper:
    """A scientific paper."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    year: str
    journal: str
    doi: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Chunk:
    """A chunk of text from a paper."""
    chunk_id: str
    paper_id: str
    text: str
    metadata: Dict


class TextChunker:
    """Chunks papers into smaller pieces."""
    
    def chunk_paper(self, paper: Paper) -> List[Chunk]:
        text = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
        
        return [Chunk(
            chunk_id=f"{paper.pmid}_0",
            paper_id=paper.pmid,
            text=text,
            metadata={
                "pmid": paper.pmid,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "journal": paper.journal,
                "doi": paper.doi
            }
        )]


# Demo neuroscience papers
DEMO_PAPERS = [
    Paper(
        pmid="demo_001",
        title="Hippocampal mechanisms of memory consolidation during sleep",
        abstract="The hippocampus plays a critical role in declarative memory consolidation. During slow-wave sleep, hippocampal sharp-wave ripples facilitate the transfer of newly encoded memories from the hippocampus to neocortical storage sites. This process involves coordinated reactivation of neural ensembles active during initial learning. Studies show that disrupting these ripples impairs memory consolidation.",
        authors=["Smith J", "Jones A", "Williams M"],
        year="2023",
        journal="Nature Neuroscience"
    ),
    Paper(
        pmid="demo_002",
        title="Adult neurogenesis in the hippocampus: functional implications",
        abstract="Neurogenesis persists in the adult hippocampus, particularly in the dentate gyrus. New neurons contribute to pattern separation, enabling discrimination of similar but distinct memories. Disruption of adult neurogenesis impairs the ability to distinguish between similar contexts and may contribute to age-related memory decline.",
        authors=["Brown K", "Davis R"],
        year="2022",
        journal="Cell"
    ),
    Paper(
        pmid="demo_003",
        title="Dopamine and reward prediction in the striatum",
        abstract="Dopaminergic neurons in the ventral tegmental area encode reward prediction errors, signaling the difference between expected and received rewards. These signals are critical for reinforcement learning and decision-making, with projections to the nucleus accumbens mediating motivated behavior.",
        authors=["Chen L", "Park S"],
        year="2023",
        journal="Neuron"
    ),
    Paper(
        pmid="demo_004",
        title="Prefrontal cortex and working memory networks",
        abstract="The dorsolateral prefrontal cortex is essential for working memory maintenance and manipulation. Persistent neural activity in prefrontal networks maintains task-relevant information over delay periods. Disruption of prefrontal function impairs cognitive flexibility and goal-directed behavior.",
        authors=["Anderson P", "Taylor S"],
        year="2021",
        journal="Journal of Neuroscience"
    ),
    Paper(
        pmid="demo_005",
        title="Synaptic plasticity mechanisms underlying learning",
        abstract="Long-term potentiation (LTP) and long-term depression (LTD) are fundamental mechanisms of synaptic plasticity. NMDA receptor activation triggers calcium influx, initiating signaling cascades that strengthen or weaken synaptic connections. These processes are essential for learning and memory formation across brain regions.",
        authors=["Martinez R", "Thompson E"],
        year="2022",
        journal="Annual Review of Neuroscience"
    ),
    Paper(
        pmid="demo_006",
        title="The role of microglia in neuroinflammation and Alzheimer's disease",
        abstract="Microglia are the primary immune cells of the central nervous system. In Alzheimer's disease, microglia become activated and cluster around amyloid plaques. While initially protective, chronic microglial activation contributes to neuroinflammation and neurodegeneration. Targeting microglial function represents a promising therapeutic approach.",
        authors=["Kim H", "Lee J", "Wang Q"],
        year="2023",
        journal="Nature Reviews Neuroscience"
    ),
    Paper(
        pmid="demo_007",
        title="Neural oscillations and cognitive processing",
        abstract="Neural oscillations at different frequencies support distinct cognitive functions. Theta rhythms (4-8 Hz) coordinate hippocampal-prefrontal communication during memory tasks. Gamma oscillations (30-100 Hz) are associated with attention and feature binding. Disrupted oscillatory dynamics are observed in various neurological disorders.",
        authors=["Wilson D", "Garcia M"],
        year="2022",
        journal="Trends in Cognitive Sciences"
    ),
]
