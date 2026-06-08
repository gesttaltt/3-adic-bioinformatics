# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

import ast
import os

BASE_DIR = "DOCUMENTATION/06_DIAGRAMS"
DIRS = [
    "01_ARCHITECTURE/models",
    "01_ARCHITECTURE/components",
    "02_SCIENTIFIC_THEORY/geometry",
    "02_SCIENTIFIC_THEORY/algebra",
    "03_WORKFLOWS/training",
    "03_WORKFLOWS/evaluation",
    "04_INFRASTRUCTURE/data",
    "04_INFRASTRUCTURE/testing",
    "04_INFRASTRUCTURE/config",
]


def ensure_dirs():
    for d in DIRS:
        path = os.path.join(BASE_DIR, d)
        os.makedirs(path, exist_ok=True)


def write_diagram(path, content):
    full_path = os.path.join(BASE_DIR, path)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Generated: {full_path}")


def get_header(title):
    return f"""%%{{init: {{'theme': 'base', 'themeVariables': {{ 'primaryColor': '#2196f3', 'edgeLabelBackground':'#f9f9f9', 'tertiaryColor': '#e1e4e8'}}}} }}%%
classDiagram
    classDef frozen fill:#e1e4e8,stroke:#333,stroke-dasharray: 5 5;
    classDef trainable fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef hyperbolic fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px;

    note "{title}"
"""


def get_graph_header(title):
    return f"""%%{{init: {{'theme': 'base', 'themeVariables': {{ 'primaryColor': '#2196f3', 'edgeLabelBackground':'#f9f9f9', 'tertiaryColor': '#e1e4e8'}}}} }}%%
graph TD
    classDef frozen fill:#e1e4e8,stroke:#333,stroke-dasharray: 5 5;
    classDef trainable fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef hyperbolic fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px;

    %% {title}
"""


def parse_imports(file_path):
    """Parses a python file to extract imported modules."""
    try:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return list(set(imports))


def parse_classes(file_path):
    """Parses a python file to extract class definitions and their methods."""
    try:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return []

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes.append({"name": node.name, "methods": methods})
    return classes


def generate_diagrams():
    ensure_dirs()

    # --- 01. ARCHITECTURE / MODELS ---

    write_diagram(
        "01_ARCHITECTURE/models/ternary_vae_v5_composition.mmd",
        get_header("V5.11 Composition")
        + """
    class TernaryVAEV5_11 {
        +FrozenEncoder encoder_A
        +FrozenEncoder encoder_B
        +DualHyperbolicProjection projection
        +DifferentiableController controller
        +FrozenDecoder decoder_A
        +forward(x)
    }
    class FrozenEncoder:::frozen {
        <<Frozen>>
        +encode(x) -> (mu, logvar)
    }
    class FrozenDecoder:::frozen {
        <<Frozen>>
        +decode(z) -> logits
    }
    class DualHyperbolicProjection:::hyperbolic {
        +forward(z_A, z_B)
    }
    class DifferentiableController:::trainable {
        +forward(stats)
    }

    TernaryVAEV5_11 *-- FrozenEncoder
    TernaryVAEV5_11 *-- FrozenDecoder
    TernaryVAEV5_11 *-- DualHyperbolicProjection
    TernaryVAEV5_11 *-- DifferentiableController
""",
    )

    write_diagram(
        "01_ARCHITECTURE/models/dual_projection_structure.mmd",
        get_header("Dual Projection")
        + """
    class DualHyperbolicProjection {
        +HyperbolicProjection proj_A
        +HyperbolicProjection proj_B
        +bool share_direction
        +forward(z_A, z_B)
    }
    class HyperbolicProjection:::hyperbolic {
        +DirectionNet direction_net
        +RadiusNet radius_net
        +forward(z) -> z_hyp
    }
    class DirectionNet:::trainable
    class RadiusNet:::trainable

    DualHyperbolicProjection *-- HyperbolicProjection
    HyperbolicProjection *-- DirectionNet
    HyperbolicProjection *-- RadiusNet
""",
    )

    write_diagram(
        "01_ARCHITECTURE/models/controller_logic.mmd",
        get_graph_header("Controller Logic")
        + """
    Input[Batch Statistics] -->|Radius, KL, Loss| Controller[DifferentiableController]:::trainable
    Controller -->|MLP| Params[Control Parameters]
    Params --> BetaA[Beta A]
    Params --> BetaB[Beta B]
    Params --> Tau[Tau Temperature]
    Params --> Weights[Loss Weights]
""",
    )

    write_diagram(
        "01_ARCHITECTURE/components/encoder_layers.mmd",
        get_graph_header("Frozen Encoder")
        + """
    Input[Input Ternary Op] -->|Flatten| L1[Linear 64]:::frozen
    L1 --> act1[SiLU]
    act1 --> L2[Linear 32]:::frozen
    L2 --> act2[SiLU]
    act2 --> Mu[Head Mu]:::frozen
    act2 --> LogVar[Head LogVar]:::frozen
""",
    )

    write_diagram(
        "01_ARCHITECTURE/components/decoder_layers.mmd",
        get_graph_header("Frozen Decoder")
        + """
    Z[Latent Z] --> L1[Linear 32]:::frozen
    L1 --> act1[SiLU]
    act1 --> L2[Linear 64]:::frozen
    L2 --> act2[SiLU]
    act2 --> Out[Output Logits]:::frozen
    Out -->|Reshape| Grid[9x3 Grid]
""",
    )

    write_diagram(
        "01_ARCHITECTURE/components/hyperbolic_internals.mmd",
        get_graph_header("Hyperbolic Projection")
        + """
    Z_euc[Euclidean Z] -->|Copy| Branch1
    Z_euc -->|Copy| Branch2

    subgraph Direction Learning
    Branch1 --> DirNet[Direction Net]:::trainable
    DirNet -->|Residual| Add[+]
    Branch1 --> Add
    Add --> Norm[Normalize L2]
    Norm --> Dir[Direction Vector]
    end

    subgraph Radial Learning
    Branch2 --> RadNet[Radius Net]:::trainable
    RadNet --> Sigmoid
    Sigmoid --> Scale[Scale by MaxRadius]
    Scale --> Rad[Scalar Radius]
    end

    Dir --> Mult[Multiply]
    Rad --> Mult
    Mult --> Z_hyp[Poincaré Z]:::hyperbolic
""",
    )

    # [Improved Auto-Generation]
    MODULE_MAP = {
        "data.generation": "src/data/generation.py",
        "data.dataset": "src/data/dataset.py",
        "models.ternary_vae": "src/models/ternary_vae.py",
        "models.differentiable_controller": "src/models/differentiable_controller.py",
        "models.homeostasis": "src/models/homeostasis.py",
        "models.hyperbolic_projection": "src/models/hyperbolic_projection.py",
        "geometry.poincare": "src/geometry/poincare.py",
        "training.trainer": "src/training/trainer.py",
        "training.monitor": "src/training/monitor.py",
        "utils.metrics": "src/utils/metrics.py",
        "utils.reproducibility": "src/utils/reproducibility.py",
        "utils.ternary_lut": "src/utils/ternary_lut.py",
        "analysis.geometry": "src/analysis/geometry.py",
        "models.curriculum": "src/models/curriculum.py",
        "losses.hyperbolic_recon": "src/losses/hyperbolic_recon.py",
        "losses.hyperbolic_prior": "src/losses/hyperbolic_prior.py",
        "losses.dual_vae_loss": "src/losses/dual_vae_loss.py",
        "training.schedulers": "src/training/schedulers.py",
        "training.environment": "src/training/environment.py",
    }

    for mod_name, file_path in MODULE_MAP.items():
        safe_mod = mod_name.replace(".", "_")

        # 1. Real Dependency Diagram
        imports = parse_imports(file_path)
        dep_content = get_graph_header(f"{mod_name} Dependencies") + f"    Center[{mod_name}]:::trainable\n"
        for imp in imports:
            if imp.startswith("src") or imp.startswith("torch") or imp.startswith("numpy") or imp.startswith("geoopt"):
                safe_imp = imp.replace(".", "_")
                dep_content += f"    Center --> {safe_imp}[{imp}]\n"

        write_diagram(f"01_ARCHITECTURE/models/module_dep_{safe_mod}.mmd", dep_content)

        # 2. Real Class Diagram
        classes = parse_classes(file_path)
        if classes:
            class_content = get_header(f"{mod_name} Structure")
            for cls in classes:
                class_content += f"    class {cls['name']} {{\n"
                for method in cls["methods"]:
                    class_content += f"        +{method}()\n"
                class_content += "    }\n"
            write_diagram(
                f"01_ARCHITECTURE/components/class_structure_{safe_mod}.mmd",
                class_content,
            )
        else:
            write_diagram(
                f"01_ARCHITECTURE/components/logic_flow_{safe_mod}.mmd",
                get_graph_header(f"{mod_name} Flow") + f"    Node[{mod_name} (Module)]:::frozen",
            )

    # Specific Detailed Diagrams
    write_diagram(
        "02_SCIENTIFIC_THEORY/geometry/poincare_distance.mmd",
        get_graph_header("Poincaré Distance Formula")
        + """
    u[Vector u]:::frozen
    v[Vector v]:::frozen
    NormU["||u||^2"]
    NormV["||v||^2"]
    Diff["||u - v||^2"]

    u --> NormU
    v --> NormV
    u --> Diff
    v --> Diff

    Term1["1 - ||u||^2"]
    Term2["1 - ||v||^2"]

    NormU --> Term1
    NormV --> Term2

    Num["2 * ||u - v||^2"]
    Diff --> Num

    Denom["Term1 * Term2"]
    Term1 --> Denom
    Term2 --> Denom

    Frac["1 + (Num / Denom)"]
    Num --> Frac
    Denom --> Frac

    Dist["arcosh(Frac)"]:::hyperbolic
    Frac --> Dist
""",
    )

    write_diagram(
        "02_SCIENTIFIC_THEORY/algebra/ternary_addition.mmd",
        get_header("Z3 Group Helper")
        + """
    class TernaryOps {
        +add(a, b)
        +mul(a, b)
    }
    class AdditionTable {
        + 0 + 0 = 0
        + 0 + 1 = 1
        + 0 + -1 = -1
        + 1 + 0 = 1
        + 1 + 1 = -1
        + 1 + -1 = 0
        + -1 + 0 = -1
        + -1 + 1 = 0
        + -1 + -1 = 1
    }
    note for AdditionTable "Modulo 3 arithmetic on balanced ternary {-1, 0, 1}"
""",
    )

    write_diagram(
        "03_WORKFLOWS/training/step_sequence.mmd",
        """
sequenceDiagram
    participant T as Trainer
    participant M as Model
    participant L as Loss
    participant O as Optimizer

    T->>M: forward(batch)
    M->>M: Encode -> Project
    M->>M: Compute Control
    M-->>T: Outputs + Control Ops
    T->>L: compute_loss(outputs, batch)
    L-->>T: Loss Dictionary
    T->>O: zero_grad()
    T->>O: backward()
    T->>O: step()
""",
    )

    write_diagram(
        "03_WORKFLOWS/training/homeostasis_loop.mmd",
        """
stateDiagram-v2
    [*] --> Monitor
    Monitor --> Check: Evaluate Metrics
    Check --> Stable: In Bounds
    Check --> Unstable: Out of Bounds
    Unstable --> Adjust: Update Controller
    Adjust --> Monitor
    Stable --> Monitor
""",
    )

    write_diagram(
        "04_INFRASTRUCTURE/testing/factory_pattern.mmd",
        get_header("Factory Test Pattern")
        + """
    class BaseFactory {
        +build(**kwargs)
        +create_batch(size)
    }
    class TernaryOperationFactory:::trainable {
        +build() -> Tensor(B, 9)
        +all_operations()
    }
    class ModelConfigFactory:::frozen {
        +minimal()
        +production()
    }

    BaseFactory <|-- TernaryOperationFactory
    BaseFactory <|-- ModelConfigFactory
""",
    )

    write_diagram(
        "04_INFRASTRUCTURE/testing/suite_map.mmd",
        get_graph_header("Testing Layout")
        + """
    Root[tests/]
    Root --> Suites[suites/]
    Root --> Core[core/]
    Root --> Factories[factories/]

    Suites --> Unit[unit/]
    Suites --> Integ[integration/]
    Suites --> E2E[e2e/]

    Core --> Builders:::trainable
    Core --> Matchers:::hyperbolic
""",
    )


if __name__ == "__main__":
    generate_diagrams()
