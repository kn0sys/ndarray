use kn0sys_ndarray::{Array, Axis};

use kn0sys_ndarray::ShapeBuilder;
use kn0sys_ndarray_rand::rand_distr::Uniform;
use kn0sys_ndarray_rand::{RandomExt, SamplingStrategy};

#[test]
fn test_dim()
{
    let (mm, nn) = (5, 5);
    for m in 0..mm {
        for n in 0..nn {
            let a = Array::random((m, n), Uniform::new(0., 2.).unwrap());
            assert_eq!(a.shape(), &[m, n]);
            assert!(a.iter().all(|x| *x < 2.));
            assert!(a.iter().all(|x| *x >= 0.));
            assert!(a.is_standard_layout());
        }
    }
}

#[test]
fn test_dim_f()
{
    let (mm, nn) = (5, 5);
    for m in 0..mm {
        for n in 0..nn {
            let a = Array::random((m, n).f(), Uniform::new(0., 2.).unwrap());
            assert_eq!(a.shape(), &[m, n]);
            assert!(a.iter().all(|x| *x < 2.));
            assert!(a.iter().all(|x| *x >= 0.));
            assert!(a.t().is_standard_layout());
        }
    }
}

#[test]
fn sample_axis_on_view()
{
    let m = 5;
    let a = Array::random((m, 4), Uniform::new(0., 2.).unwrap());
    let _samples = a
        .view()
        .sample_axis(Axis(0), m, SamplingStrategy::WithoutReplacement);
}

#[test]
#[should_panic]
fn oversampling_without_replacement_should_panic()
{
    let m = 5;
    let a = Array::random((m, 4), Uniform::new(0., 2.).unwrap());
    let _samples = a.sample_axis(Axis(0), m + 1, SamplingStrategy::WithoutReplacement);
}

#[test]
#[should_panic]
fn sampling_without_replacement_from_a_zero_length_axis_should_panic()
{
    let n = 5;
    let a = Array::random((0, n), Uniform::new(0., 2.).unwrap());
    let _samples = a.sample_axis(Axis(0), 1, SamplingStrategy::WithoutReplacement);
}

#[test]
#[should_panic]
fn sampling_with_replacement_from_a_zero_length_axis_should_panic()
{
    let n = 5;
    let a = Array::random((0, n), Uniform::new(0., 2.).unwrap());
    let _samples = a.sample_axis(Axis(0), 1, SamplingStrategy::WithReplacement);
}
