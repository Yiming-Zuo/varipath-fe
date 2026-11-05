import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from varipath.analytical import _derivatives, _plateau_map, _route, run_analytical


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    import torch
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    path=tmp_path_factory.mktemp("e0")/"result.json"
    result=run_analytical(path)
    torch.set_num_threads(previous)
    assert json.loads(path.read_text())==result
    return result


def test_e0_full_control_set_and_native_finite_maps(report):
    assert report["status"]=="passed"
    assert len(report["gates"])==11
    assert all(report["gates"].values())
    assert len(report["continuous_responses"])==6
    assert len(report["finite_step_responses"])==6
    assert set(report["negative_controls"])=={
        "A_invariant_boundary","B_irrelevant_true_saddle",
        "C_entropic_bottleneck","D_fitted_endpoint",
    }
    assert report["native_finite_map"]["n_steps"]==32
    assert report["native_finite_map"]["conjugated_logweight_error"]<1e-12
    assert report["native_finite_map"]["absolute_endpoint_separation"]>1e-6
    assert report["quadrature_order64_order96_maximum_change"]<2e-8


def test_same_side_finite_effects_and_fixed_boundary_shape(report):
    first,second=report["same_side_pair"]
    assert first["finite_mass_change"]>0 and second["finite_mass_change"]>0
    assert first["finite_risk_change"]>0 and second["finite_risk_change"]<0
    assert max(abs(row["mass_half_dose_residual"]) for row in (first,second))<1e-12
    assert max(abs(row["risk_half_dose_residual"]) for row in (first,second))<1e-9
    invariant=report["negative_controls"]["A_invariant_boundary"]
    assert abs(invariant["measured_mass_change"])<1e-12
    assert invariant["measured_conditional_h2"]>1e-8
    assert invariant["measured_conditional_h2"]==pytest.approx(invariant["predicted_conditional_h2"],rel=.002)
    fitted=report["negative_controls"]["D_fitted_endpoint"]
    assert fitted["risk_before"]==0
    assert fitted["risk_full"]>fitted["risk_half"]>0
    assert fitted["measured_quadratic_coefficient"]==pytest.approx(fitted["predicted_quadratic_coefficient"],rel=.002)


def test_flux_matches_independent_one_dimensional_formula():
    sigma,c=.6,-.5
    normalization=quad(lambda x:math.exp(-(x*x-1)**2),-5,5,epsabs=1e-13)[0]
    integral=quad(lambda t:t*(1-t)*t*t*(1-t)*(1-t)*math.exp(-c*c*(t*t*(1-t)*(1-t))**2/(2*sigma*sigma)),0,1,epsabs=1e-14)[0]
    expected=-.5*c*math.exp(-1)*integral/(sigma*sigma*normalization*math.sqrt(1+1/(sigma*sigma)))
    prediction=_derivatives(c,sigma)
    assert prediction["mass"]==pytest.approx(expected,abs=2e-12)
    assert abs(prediction["score_mean"])<1e-11


def test_plateau_specialization_rejects_off_domain_and_native_dose_bound():
    from varipath.interventions import InterventionError
    with pytest.raises(ValueError,match="非平台"):
        _plateau_map(_route(.5,epsilon=.01),np.array([[40.,0.]]))
    with pytest.raises(InterventionError,match="收缩界"):
        _route(.5,epsilon=1000.)
