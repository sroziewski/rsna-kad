import numpy as np
import pytest

from conftest import STUDY_A, STUDY_B, UID_ROOT, SyntheticData
from rsna_kad import dicom


def test_fixture_generated_fast(synthetic: SyntheticData):
    assert synthetic.seconds < 5.0, f"synthetic study took {synthetic.seconds:.1f}s"
    assert len(synthetic.series_df()) == 9
    assert len(synthetic.spec(UID_ROOT + "200002").paths) == 120


def test_read_header_fields(synthetic: SyntheticData):
    spec = synthetic.spec(UID_ROOT + "200003")
    h = dicom.read_header(spec.paths[0])
    assert (h.study_uid, h.series_uid) == (STUDY_A, spec.uid)
    assert (h.rows, h.cols) == (640, 1280)
    assert h.spacing == pytest.approx(0.25)
    assert h.manufacturer == "Philips Medical Systems"
    assert h.rescale_slope == pytest.approx(4.5 + 0.01 * (h.instance_number - 1))
    assert h.series_description == "COR T1 L"
    assert h.laterality == ""
    assert h.series_number == 3


def test_truncated_slice_returns_none(synthetic: SyntheticData):
    spec = synthetic.spec(UID_ROOT + "200001")
    paths = dicom.sort_series(spec.paths)
    assert dicom.read_slice(paths[10]) is None
    good = dicom.read_slice(paths[11])
    assert good is not None and good.shape == (512, 512) and good.dtype == np.float32


def test_read_slice_applies_philips_rescale(synthetic: SyntheticData):
    import pydicom
    spec = synthetic.spec(UID_ROOT + "200003")
    path = dicom.sort_series(spec.paths)[0]
    raw = pydicom.dcmread(path).pixel_array.astype(np.float32)
    out = dicom.read_slice(path)
    assert np.allclose(out, raw * 4.5, atol=1e-2)
    assert out.max() > 3000


def test_read_slice_keeps_signed_values(synthetic: SyntheticData):
    spec = synthetic.spec(UID_ROOT + "200004")
    out = dicom.read_slice(spec.paths[0])
    assert out.min() < -500
    assert out.max() > 500


def test_sort_series_matches_ipp_order(synthetic: SyntheticData):
    for uid in (UID_ROOT + "200001", UID_ROOT + "200002", UID_ROOT + "200003"):
        spec = synthetic.spec(uid)
        paths = dicom.sort_series(spec.paths)
        headers = [dicom.read_header(p) for p in paths]
        numbers = [h.instance_number for h in headers]
        assert numbers == list(range(1, spec.n + 1))
        assert dicom.is_monotone_order(headers)
        pos = dicom.slice_positions(headers)
        assert np.allclose(np.diff(pos), spec.z_step) or np.allclose(np.diff(pos), -spec.z_step)


def test_series_geometry(synthetic: SyntheticData):
    spec3d = synthetic.spec(UID_ROOT + "200002")
    g = dicom.series_geometry([dicom.read_header(p) for p in dicom.sort_series(spec3d.paths)])
    assert g.plane == "Axial" and g.is_3d and g.n_slices == 120
    assert g.z_step == pytest.approx(0.9)
    assert g.spacing == pytest.approx(0.47)
    assert g.fov_mm == pytest.approx(256 * 0.47)
    assert g.ipp_centre[0] == pytest.approx(60.0, abs=1e-6)

    spec2d = synthetic.spec(UID_ROOT + "200001")
    g2 = dicom.series_geometry([dicom.read_header(p) for p in dicom.sort_series(spec2d.paths)])
    assert g2.plane == "Sagittal" and not g2.is_3d and g2.z_step == pytest.approx(3.0)
    assert g2.fov_mm == pytest.approx(204.8)

    specc = synthetic.spec(UID_ROOT + "200003")
    gc = dicom.series_geometry([dicom.read_header(p) for p in dicom.sort_series(specc.paths)])
    assert gc.plane == "Coronal" and (gc.rows, gc.cols) == (640, 1280)


def test_is_3d_from_slice_count_when_tag_absent():
    from rsna_kad.dicom import Header
    def mk(i, acq):
        return Header("p", "s", "r", str(i), i, 1, 64, 64, 0.5, (1, 0, 0, 0, 1, 0), (0, 0, i * 1.0),
                      1.0, 1.0, 1.0, 0.0, 0, 12, "GE", "", "", "", acq)
    assert dicom.series_geometry([mk(i, "") for i in range(70)]).is_3d
    assert not dicom.series_geometry([mk(i, "") for i in range(40)]).is_3d
    assert not dicom.series_geometry([mk(i, "2D") for i in range(70)]).is_3d


def test_plane_of_iop():
    assert dicom.plane_of_iop((0, 1, 0, 0, 0, -1)) == "Sagittal"
    assert dicom.plane_of_iop((1, 0, 0, 0, 0, -1)) == "Coronal"
    assert dicom.plane_of_iop((1, 0, 0, 0, 1, 0)) == "Axial"
    oblique = (0.98, 0.17, 0.05, -0.17, 0.98, 0.05)
    assert dicom.plane_of_iop(oblique) == "Axial"


def test_study_b_files_exist(synthetic: SyntheticData):
    assert synthetic.series_dir(STUDY_B, UID_ROOT + "300001").is_dir()
    assert len(dicom.list_series_files(synthetic.series_dir(STUDY_B, UID_ROOT + "300001"))) == 12
