use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyTuple;
use numpy::IntoPyArray;
use ndarray::{Array2, ArrayD};
use fitsio::FitsFile;
use std::path::Path;
use rayon::prelude::*;
use fitsio::hdu::HduInfo;
use ndarray::Ix2;


/// Read a FITS file and return its image as an Array2<f32>
fn read_fits_image(path: &Path) -> Result<(Array2<f32>, Option<f32>, Option<f64>), Box<dyn std::error::Error + Send + Sync>> {
    let mut fits = FitsFile::open(path)?;
    let hdu = fits.primary_hdu()?;

    // Try to read the "LBT_PARA" keyword from the header
    let lbt_para: Option<f32> = hdu
        .read_key::<String>(&mut fits, "LBT_PARA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok());
    
    // Try to read the "PCJD" keyword from the header
    let timestamp: Option<f64> = hdu
        .read_key::<String>(&mut fits, "PCJD")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());

    let vec_data: Vec<f32> = hdu.read_image(&mut fits)?;


    let info = hdu.info;
    let shape_vec = match info {
        HduInfo::ImageInfo { shape, .. } => shape,
        _ => return Err("Primary HDU is not an image".into()),
    };

    let shape_dyn = ndarray::IxDyn(&shape_vec);
    let arrayd = ArrayD::from_shape_vec(shape_dyn, vec_data)?;


    let array2 = match arrayd.ndim() {
        2 => arrayd.into_dimensionality::<Ix2>()?,
        3 => {
            let view = arrayd
                .index_axis(ndarray::Axis(0), arrayd.shape()[0] - 1)
                .to_owned();
            view.into_dimensionality::<Ix2>()?
        }
        _ => return Err(format!("Unsupported image dimensions: {:?}", arrayd.shape()).into()),
    };

    Ok((array2, lbt_para, timestamp))
}

/// Compute mean image from list of paths
fn compute_mean_image(paths: &[String]) -> Result<Array2<f32>, Box<dyn std::error::Error + Send + Sync>> {
    // Parallel map + reduce
    let (sum, count) = paths
        .par_iter()
        .map(|path| {
            let (img, _, _) = read_fits_image(Path::new(path))?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((img, 1f32))
        })
        .reduce(
            || Ok((Array2::<f32>::zeros((0, 0)), 0f32)),
            |a, b| {
                let (a_img, a_count) = a?;
                let (b_img, b_count) = b?;

                // If one of the images is zero-sized (the identity), return the other
                if a_img.is_empty() {
                    return Ok((b_img, b_count));
                }
                if b_img.is_empty() {
                    return Ok((a_img, a_count));
                }

                let sum = a_img + &b_img;
                Ok((sum, a_count + b_count))
            },
        )?;

    if count == 0.0 {
        return Err("No images to average".into());
    }

    Ok(sum / count)
}


/// Subtract mean image from each image in list_a
#[pyfunction]
fn subtract_mean_from_list<'py>(
    py: Python<'py>,
    list_a: Vec<String>,
    list_b: Vec<String>,
) -> PyResult<&'py PyList> {
    let mean_img = compute_mean_image(&list_b)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

// Process each file in list_a in parallel
    let results: Result<Vec<(Array2<f32>, f32, f64)>, Box<dyn std::error::Error + Send + Sync>> =
        list_a.par_iter()
            .map(|path| {
                let (img, rotation, jd) = read_fits_image(Path::new(path))?;
                let subtracted = img - &mean_img;
                Ok((subtracted, rotation.unwrap_or(0.0), jd.unwrap_or(0.0)))
            })
            .collect();

    let results = results.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Convert each (image, rotation, jd) into a Python tuple
    let py_results = PyList::new(
        py,
        results.into_iter().map(|(img, rot, jd)| {
            PyTuple::new(
                py,
                &[
                    img.into_pyarray(py).into_py(py),
                    rot.into_py(py),
                    jd.into_py(py),
                ],
            )
        }),
    );

    Ok(py_results)
}

/// Python module definition
#[pymodule]
fn fits_lizard(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(subtract_mean_from_list, m)?)?;
    Ok(())
}
