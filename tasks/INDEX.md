# GS-GHOST Task Index

## Build Order

| Order | Task | Summary | Depends |
|------|------|---------|---------|
| 1 | [PRD-0101](./PRD-0101.md) | Normalize FUJIN scaffold names to GS-GHOST | None |
| 2 | [PRD-0102](./PRD-0102.md) | Add typed config and settings models | PRD-0101 |
| 3 | [PRD-0103](./PRD-0103.md) | Implement asset and layout registry | PRD-0102 |
| 4 | [PRD-0104](./PRD-0104.md) | Add preflight config/layout tests | PRD-0103 |
| 5 | [PRD-0201](./PRD-0201.md) | Implement SAM2 and SfM wrapper interfaces | PRD-0104 |
| 6 | [PRD-0202](./PRD-0202.md) | Implement prior retrieval and prompt flow | PRD-0201 |
| 7 | [PRD-0203](./PRD-0203.md) | Implement prior-mask affine alignment | PRD-0202 |
| 8 | [PRD-0204](./PRD-0204.md) | Implement HaMeR postprocessing and jitter rejection | PRD-0201 |
| 9 | [PRD-0205](./PRD-0205.md) | Implement grasp detection and HO alignment losses | PRD-0203, PRD-0204 |
| 10 | [PRD-0206](./PRD-0206.md) | Implement object and hand Gaussian optimization modules | PRD-0205 |
| 11 | [PRD-0301](./PRD-0301.md) | Build sequence manifest and frame ingestion CLI | PRD-0206 |
| 12 | [PRD-0302](./PRD-0302.md) | Orchestrate preprocessing stage runner | PRD-0301 |
| 13 | [PRD-0303](./PRD-0303.md) | Orchestrate object and combined training stages | PRD-0302 |
| 14 | [PRD-0304](./PRD-0304.md) | Export viewer assets and artifact manifest | PRD-0303 |
| 15 | [PRD-0401](./PRD-0401.md) | Encode ARCTIC and HO3D benchmark definitions | PRD-0304 |
| 16 | [PRD-0402](./PRD-0402.md) | Implement 2D rendering evaluation | PRD-0401 |
| 17 | [PRD-0403](./PRD-0403.md) | Implement ARCTIC export and 3D metric bridge | PRD-0401 |
| 18 | [PRD-0404](./PRD-0404.md) | Generate paper comparison report | PRD-0402, PRD-0403 |
| 19 | [PRD-0501](./PRD-0501.md) | Define FastAPI schemas and app skeleton | PRD-0304 |
| 20 | [PRD-0502](./PRD-0502.md) | Wrap pipeline in background job runner | PRD-0501 |
| 21 | [PRD-0503](./PRD-0503.md) | Containerize the service and add smoke tests | PRD-0502 |
| 22 | [PRD-0601](./PRD-0601.md) | Define ROS2 contracts and node shell | PRD-0503 |
| 23 | [PRD-0602](./PRD-0602.md) | Implement sequence dispatch from ROS2 | PRD-0601 |
| 24 | [PRD-0603](./PRD-0603.md) | Add launch files and ROS2 regression tests | PRD-0602 |
| 25 | [PRD-0701](./PRD-0701.md) | Build production preflight checks | PRD-0404, PRD-0503 |
| 26 | [PRD-0702](./PRD-0702.md) | Add reproducibility manifests and packaging | PRD-0701 |
| 27 | [PRD-0703](./PRD-0703.md) | Final production validation and release checklist | PRD-0702 |

## Notes
- Follow the tasks in order unless a task is explicitly marked independent.
- Each task is scoped to one focused coding session.
- The paper-faithful core ends at PRD-0404. API, ROS2, and production are ANIMA wrappers on top.

