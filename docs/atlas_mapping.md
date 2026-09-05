# Phase 3 atlas mapping

## Selection and provenance

NeuroSplit uses the **Destrieux 2009** (`aparc.a2009s`) anatomical cortical
parcellation for Phase 3. It is a sulco-gyral parcellation described by
Destrieux et al. (2010), distributed with FreeSurfer, and loaded through
`nilearn.datasets.fetch_atlas_surf_destrieux`.

- Surface: `fsaverage5`
- Hemisphere behavior: separate left and right annotations
- Annotation labels: 75 per hemisphere excluding `Unknown`
- Cortical analytical ROIs: 148 (74 per hemisphere)
- Medial wall: atlas label 42, excluded; 888 left + 881 right = 1,769 vertices
- Source license: reported as **unknown** by nilearn. NeuroSplit deliberately
  does not claim a license that the source does not establish.

Scientific reference: Destrieux C, Fischl B, Dale A, Halgren E. *Automatic
parcellation of human cortical gyri and sulci using standard anatomical
nomenclature.* NeuroImage 53(1), 2010. DOI: 10.1016/j.neuroimage.2010.06.010.

## Why ordering matters, and how it was verified

An atlas is an array: label at position `j` applies to surface vertex `j`.
Matching lengths cannot prove matching order. A permutation would still have
10,242 values and would silently assign responses to the wrong anatomy.

TRIBE declares fsaverage5 and emits 10,242 left vertices followed by 10,242
right vertices. For each hemisphere, NeuroSplit compares the exported Phase 2
pial coordinates against nilearn's atlas-source fsaverage5 coordinates with
exact array equality. The observed maximum coordinate delta is 0.0. Loading is
refused if any coordinate differs. Only then are left and right atlas labels
concatenated in prediction order.

The persisted dense `vertex_to_region` array has length 20,484. Values 0–147
identify hemisphere-specific regions; `-1` identifies excluded medial-wall or
otherwise unmapped vertices. Region vertex lists are not duplicated in JSON.

## Validation result

For the checked-in fsaverage5 mesh:

| Quantity | Count |
|---|---:|
| Prediction vertices | 20,484 |
| Mesh vertices | 20,484 |
| Atlas vertices | 20,484 |
| Mapped cortical vertices | 18,715 |
| Medial-wall vertices | 1,769 |
| Unknown non-wall vertices | 0 |
| Invalid labels | 0 |

Mapped fraction is 91.365%; the excluded 8.635% is the explicit medial wall,
not missing data.

## Functional networks

Phase 3 does not currently emit network analytics. No Yeo/Schaefer annotation
was locally verified against the exact exported ordering during this phase.
Network names or cross-atlas assignments are never inferred manually. A future
network implementation must pass the same coordinate/order, hemisphere, and
medial-wall validation.
