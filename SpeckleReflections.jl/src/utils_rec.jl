# struct CropParams
#    crop_size::NTuple{2, Int}  # size to crop camera Image [pixel]
#    crop_center::NTuple{2, Int}# center of the region to be cropped [pixel index]
# end

# Define iluination modes (Pure datatypes to be fast)
abstract type IlluMode end  # Encompassing type
struct Plane <: IlluMode end  # Type to generate a plane wave illumination
struct Speckle <: IlluMode end # Type ot generate a spackle illumination

abstract type CalibMode end  # Encompassing type
struct CalibDivide <: CalibMode end  # Type to generate a plane wave illumination
struct CalibConj <: CalibMode end # Type ot generate a spackle illumination
struct CalibPhase <: CalibMode end # Type ot generate a spackle illumination

"""
struct HoloParams
   cent::NTuple{2, Int}    # Center of 1.Order in k-space, [pixel Index]
   pup_dia::NTuple{2, Int} # pupil diameter of 1. Order in k-Pscae, [pixel]
   pad_size::NTuple{2, Int}# size to upsample to [pixel]
   pad_fac::Int            # upsampling factor, needs to corespond to pad_size
   preserve_size::Bool     # if true, the size of the reconstruction agrees to the raw data size
   double_size::Bool       # if true (and not preserve_size), twice the size of pup_dia is used to crop.
"""
struct HoloParams
   cent::NTuple{2, Int}    # Center of 1.Order in k-space, [pixel Index]
   pup_dia::NTuple{2, Int} # pupil diameter of 1. Order in k-Pscae, [pixel]
   pad_size::NTuple{2, Int}# size to upsample to [pixel]
   pad_fac::Int            # upsampling factor, needs to corespond to pad_size
   preserve_size::Bool     # if true, the size of the reconstruction agrees to the raw data size
   double_size::Bool       # if true (and not preserve_size), twice the size of pup_dia is used to crop.
   calib_mode::Type
end

"""
struct ImageSettings
   lam::Float64            # wavelength [um]
   n::Float64        # refractive Index of medium
   NA_illu::Float64   # numerical aperture of the speckle illumination
   NA_det::Float64   # numerical aperture of the detection optics
   k_ref::NTuple{2, Float64} # position of the reference wave in Fourier space in relative frequencies border is (0.5, 0.5)
   mag::Int                # Magnification of the Objective
   samp::NTuple{3, Float64} # Camera-Pixelsize um on sampel
   ref_plane::Int            # Index of the plane of the 3D set where the 2D data was recorded
   illu_mode::Type       # Mode of illumination (e.g. Speckle)
   # Npx::NTuple{2, Int}     # Nbr of pixels in x and y on Camera
   # Nimags::Int             # Nbr. of images recorded
end
"""
struct ImageSettings
   lam::Vector{Float64}          # wavelength [um]
   n::Float64        # refractive Index of medium
   NA_illu::Float64
   NA_det::Float64
   mag::Int                # Magnification of the Objective
   samp::NTuple{3, Float64} # Camera-Pixelsize um on sampel
   illu_mode::Type       # Mode of illumination (e.g. Speckle)
   nbr_cam_pixels::NTuple{2, Int}
end

# A function to only collect non-basic array types. To ensure that CuArrays are Not removed from the GPU!
opt_collect(a) = collect(a)

normalize_intensity(input) = input ./ sum(input)

"""
   reconstruct_speckle_hologram!(res_obj, res_illu, data::AbstractArray, holo_params::HoloParams, settings::ImageSettings, Nz, rec_field_illu=nothing; translation_px=nothing, ref=nothing, NA = 0.9)

Main Method to reconstruct a single off axis Speckle Hologram.

Parameters:
- res_obj, res_illu: temp array into which to write the results
- data: input data, a stack a speckle data.
      If this is already complex-valued, the already extracted holograms in Fourier space are assumed to be given.
      Otherwise real data assumes data from which the holograms still have to be extracted.
- holo_params: collection of parameters used for the reconstruction. See `get_holo_params`.
- settings:  see `get_settings` and `get_settings_meas`
"""
function reconstruct_speckle_hologram!(calib_trans, calib_illu, data::AbstractArray, holo_params::HoloParams, settings::ImageSettings, Nz, lam, rec_field_illu_ft=nothing; translation_px=nothing, ref=nothing)

   # 2. reconstruct single-plane off-axis Hologramm
   rec_field_obj_ft = (eltype(data) <: Complex) ? data : opt_collect(rec_holo_shift(data, holo_params, ref))

   new_sampling = ((settings.samp[1:2] .* size(data)[1:2] ./ size(rec_field_obj_ft)[1:2])..., settings.samp[3])

   PType = complex_arr_type(typeof(expand_dims(rec_field_obj_ft, Val(3))))
   propagator = propagator_col(PType, (size(rec_field_obj_ft)[1:2]..., Nz), new_sampling, lam; ref_idx = settings.ref_plane)

   aperture = get_propagator_mask_3d(rec_field_obj_ft, lam, new_sampling[1:2], settings.NA_det)

   rec_field_obj_prop = ift2d(rec_field_obj_ft .* conj.(propagator .* aperture));

   rec_field_illu_ft = select_region_view(rec_field_illu_ft, size(rec_field_obj_ft))
   rec_field_illu_prop = ift2d(rec_field_illu_ft .* opt_collect(propagator .* aperture));
  
   # 4. upsample results
   nsz = size(calib_trans)

   rec_field_illu_prop = (holo_params.pad_fac > 0 && !isnothing(rec_field_illu_prop)) ? resample(opt_collect(rec_field_illu_prop), nsz) : rec_field_illu_prop

   rec_field_obj_prop = (holo_params.pad_fac > 0 && !isnothing(rec_field_obj_prop)) ? resample(opt_collect(rec_field_obj_prop), nsz) : rec_field_obj_prop

   # 5. calibrate Hologramm -> make stand alone method to calibrate, i.e. divide by the illumination field
   calib_trans .= isnothing(rec_field_illu_prop) ? rec_field_obj_prop :
                  calibrate_hologram(rec_field_obj_prop, illu_holo=rec_field_illu_prop, calib_mode=holo_params.calib_mode)

   calib_illu .= isnothing(rec_field_illu_prop) ? rec_field_illu_prop :
                  calibrate_hologram(rec_field_illu_prop, illu_holo=rec_field_illu_prop, calib_mode=holo_params.calib_mode)
   # 6. back translate reconstructed sampel images according translation matrix and mask outher regions
   if !isnothing(translation_px)
      calib_trans .= shift(calib_trans, translation_px)
   end

   return  calib_trans, calib_illu # convert(Array{ComplexF64}, SD_calib_trans)
end

function reconstruct_plane_hologram!(calib_trans, data::AbstractArray, holo_params::HoloParams, settings::ImageSettings, Nz, lam; translation_px=nothing, ref=nothing)
   rec_field_obj_ft = (eltype(data) <: Complex) ? data : opt_collect(rec_holo_shift(data, holo_params, ref))

   # rec_field_obj = opt_collect(rec_field_obj)
   # rec_field_propagator = propagate_field_3d_r(rec_field_obj, settings.lam, settings.samp; pad_factor = 1, Nz=length(prop_dist))

   new_sampling = ((settings.samp[1:2] .* size(data)[1:2] ./ size(rec_field_obj_ft)[1:2])..., settings.samp[3])

   PType = complex_arr_type(typeof(expand_dims(rec_field_obj_ft, Val(3))))
   propagator = propagator_col(PType, (size(rec_field_obj_ft)[1:2]..., Nz), new_sampling, lam; ref_idx = 1)

   aperture = get_propagator_mask_3d(rec_field_obj_ft, lam, new_sampling[1:2], settings.NA_det)

   rec_field_obj_prop = ift2d(rec_field_obj_ft .* conj.(propagator .* aperture)); # keep in pupil plane for further oct reconstruction

   # 4. upsample results
   nsz = size(calib_trans)#size(rec_field_obj_prop)[2].*(holo_params.pad_fac, holo_params.pad_fac)
   # if (ndims(rec_field_obj_prop) > 2)
   #    nsz = (nsz..., size(rec_field_obj_prop)[3])
   # end

   calib_trans .= (holo_params.pad_fac > 0 && !isnothing(rec_field_obj_prop)) ? resample(opt_collect(rec_field_obj_prop), nsz) : rec_field_obj_prop
end

function calibrate_hologram(obj_holo; illu_holo=nothing, calib_mode::Type = CalibConj)
   if isnothing(illu_holo)
      return obj_holo
   end
   if (calib_mode == CalibConj)
      return obj_holo .* conj.(illu_holo)
   elseif (calib_mode == CalibPhase)
      return obj_holo .* exp.(-1im .* angle.(illu_holo))
   elseif (calib_mode == CalibDivide)
      ave_mag = isnothing(illu_holo) ? 1 : sum(abs.(illu_holo))/(size(illu_holo,1)*size(illu_holo,2)) # find average Intensity in illumination and define regularisation strength on it
      eps = ave_mag;
      return obj_holo .* conj.(illu_holo) ./ (illu_holo.*conj(illu_holo) .+ eps)
   else
      error("unknown calibration mode. Choose CalibConj, CalibPhase or CalibDivide")
   end
end

"""
   get_rel_ref(sz::NTuple, abs_pos)

calculates the relative frequency position of an absolute pixel position in Fourier space
"""
function get_rel_ref(sz::NTuple{2, Int}, abs_pos)
   f_mid = sz.÷2 .+1
   rel_ref = (abs_pos .- f_mid) ./ sz
   return rel_ref
end

"""
   get_rel_ref(img, abs_pos)

calculates the relative frequency position of an absolute pixel position in Fourier space
"""
function get_rel_ref(img::AbstractArray, abs_pos)
   return get_rel_ref(size(img)[1:2], abs_pos)
end

"""
get_settings()

Method that defines the parameters which were used in the setup with which the Images were captured.
Should be adaptet in the future to read those out of a file.
"""
function get_settings(;lam, n=1.0, NA_illu=0.16, NA_det=0.16,  mag=5, samp_z = 0.1, samp = (3.9/mag, 3.9/mag, samp_z), illu_mode=Plane, nbr_cam_pixels=(4704, 3424))
    # To DO: store parameters in file which is automatically generated when measurements are saved
    # settings for simulation
    settings = ImageSettings(lam, n, NA_illu, NA_det, mag, samp, illu_mode, nbr_cam_pixels)
    return settings
end

function get_settings(settings;lam=settings.lam, n=settings.n, NA_illu=settings.NA_illu, NA_det=settings.NA_det, mag=settings.mag, samp = settings.samp, illu_mode=settings.illu_mode, nbr_cam_pixels=settings.nbr_cam_pixels)
    # To DO: store parameters in file which is automatically generated when measurements are saved
    # settings for simulation
    return ImageSettings(lam, n, NA_illu, NA_det, mag, samp, illu_mode, nbr_cam_pixels)
end

function get_settings_meas(;lam, n=1.0, NA_illu=0.16, NA_det=0.16, samp_z=0.5, mag=5, samp = (2.5/mag, 2.5/mag, samp_z), illu_mode=Speckle, nbr_cam_pixels=(5120, 5120))
    # To DO: store parameters in file which is automatically generated when measurements are saved
    # settings for measurements
    return get_settings(;lam, n=n, NA_illu=NA_illu, NA_det=NA_det, samp=samp, samp_z=samp_z, mag=mag, illu_mode=illu_mode, nbr_cam_pixels=nbr_cam_pixels)
end

"""
    get_holo_params(sz::NTuple{2,Int}, pad_fac, settings::ImageSettings; do_crop)

Method that reads the coordinates used in k-space to crop the first order from a txt file
and returns them as a HoloParams Struct

# Arguments
+ crop_params: struct that holds the real space image crop püarameters
+ pad_fac: the facor with which the reconstructed image should be upsampeled
+ settings: An instance of ImageSettings struct that holds all relevant parameters which describe the physical setup with which the image was captured
+ do_crop: boolean to indicate if the real space image will be cropped or not
"""
function get_holo_params(sz::NTuple{2, Int}, pad_fac; cent=nothing, pup_dia=nothing, preserve_size=false, double_size=false, calib_mode::Type = CalibConj)
   cent = isnothing(cent) ? round.(sz[1:2]./2) : cent
   pup_dia = isnothing(pup_dia) ? round.(sz[1:2]./2) : pup_dia

   pad_size = sz .* pad_fac
   if length(pad_size) >= 3
      sz = collect(pad_size)
      sz[3,] = 1
      pad_size = tuple(sz)
   end

   return HoloParams(cent .+ 1, pup_dia, pad_size, pad_fac, preserve_size, double_size, calib_mode)
end

function get_holo_params(data::AbstractArray, pad_fac; kwargs...)
   return get_holo_params(size(data), pad_fac; kwargs...)
end

"""
   fd_zeropad(SD, pad_size)

Simpel method that zeropads the given spatial domain field in frequency domain
and returns the zeropadded SPatial domain field

# Arguments
+ SD: A complex array representing the input field in the spatial domain to be interpolated
+ pad_size: The total size, the returned field should have
"""
function fd_zeropad(SD, pad_fac)
   FD = ft2d(SD)
   pad_size = size(SD).*pad_fac;
   pad_size = (iseven(pad_fac) | (pad_fac==1)) ? pad_size : pad_size = pad_size .-1;
   #FD_win = FD.*hanning(size(FD))
   SD = ift2d(select_region(FD, new_size=pad_size))
   return SD # convert(Array{ComplexF64},SD);
end

# """
#    rec_holo_cut(holo, holo_params, pad_size)

# Reconstructs an off-axis hologram by selecting a specific region around the 1. Order Diffraction in the Fourier Transform
# The function crops the Fourier Transform of the given off-axis hologram to the specified pupil diameter, with provided
# center Coordinates of the 1. Order Diffraction and applies zero-padding before performing the inverse Fourier Transform.

# The result remains to be in Fourier space, but you can apply `ift2d(res)` to investigate the real space features.

# # Arguments
# + holo: The input hologram to be reconstructed
# + holo_params: parameters for hologram reconstruction
#    + center: The center point of the 1. order diffraction in the Fourier Domain
#    + pup_dia: The diameter of the pupil of the 1. diffraction
# + pad_size: The size to which the selected region will be zero-padded before the inverse transform
# """
# function rec_holo_cut(holo, holo_params::HoloParams, pad_size=nothing)
#    # shift by cropping region of FT image
#    FD = select_region(ft2d(holo), new_size=holo_params.pup_dia, center=holo_params.cent)
#    # mask outher regions of 1. Order
#    preserve_center_only!(FD)
#    # zeropad and Fourier Transform back to spatial Domain
#    if (holo_params.preserve_size)
#       select_region(FD, new_size=pad_size)
#    else
#       if (holo_params.double_size)
#          select_region(FD, new_size=2 .* holo_params.pup_dia)
#       else
#          select_region(FD, new_size=holo_params.pup_dia)
#       end
#    end
#    return FD # convert(Array{ComplexF64},SD);
#    # SD = ift2d(FD)
#    # return SD # convert(Array{ComplexF64},SD);
# end

"""
   rec_holo_shift(holo, holo_params::HoloParams, ref, real_space_shift=nothing)

reconstruct a hologram by shifting with the reference wave, Fourier transforming and clipping.
The parameter `holo_params.pup_dia` defines the pupil diameter in pixels to clip at.
A round mask is also applied.
The result is still in Fourier space!

# Parameters
- holo:  The hologram data to process
- holo_params: The parameters for the hologram. `holo_params.pup_dia` defines the pupil diameter in pixels to clip at.
   A bounding box is automatically applied
   - holo_params.preserve_size: If true, the output will have the same pixel number and sampling as the input (not recommended!)
   - holo_params.double_size: If `preserve_size` is `false`, the output size will be double the clipped size.
If both `preserve_size` and `double_size`are `false`, the cropped size according to `holo_params.pup_dia` is returned.
Note that this will undersample intensities, once calculated.
"""
function rec_holo_shift(holo, holo_params::HoloParams, ref, real_space_shift=nothing)
   shifter = isnothing(real_space_shift) ? 1 : exp_ikx_sep(size(holo)[1:2]; shift_by=real_space_shift)
   FD = select_region(ft2d(holo .* ref) .* shifter, new_size=holo_params.pup_dia) # Note that padding in real space does NOT affect real space sampling
   preserve_center_only!(FD);
   if (holo_params.preserve_size)
      return select_region(FD, new_size=size(holo))
   else
      if (holo_params.double_size)
         return select_region(FD, new_size=holo_params.pup_dia)
         # return ift2d(select_region(FD, new_size=holo_params.pup_dia))
      else
         return FD
         #return ift2d(FD)
      end
   end
end

"""
   preserve_center_only!(holo, sz)

sets the outer part of an array (holo) to zero, sz denoting the size of the region to preserve.
sz is required to be odd.
"""
function preserve_center_only!(holo)
   # mask = disc(size(holo)[1:2])
   holo .= holo .* (rr2_sep(real_arr_type(typeof(holo)), size(holo)[1:2], scale = 2 ./ size(holo)[1:2]) .< 1)
end

"""
   get_propagator(field, distance, lam, samp)

returns the propagator a `field` is mulitiplied (in Fourier-space) by to propagate by `distance` along the axial direction
using the wavelength `lam` and a sampling `samp`. Not that only the XY sampling is considered.
The obtained propagator is centered in Fourier space according the the convertion of the `ft` command.
Note that the propagator only returns the complex phase part but not the amplitude part.
"""
function get_propagator(field, distance, lam, samp)
    if length(samp)> 2
       error("only two-dimensional sampling allowed. Use samp[1:2]")
    end
    sz_xy = size(field);
    @assert samp[1] == samp[2]
    return propagator_col(typeof(field), sz_xy, (samp[1], samp[2], distance), lam); # RH
    # return propagator_col(typeof(field), sz_xy; Δz=distance./lam, k_max=samp[1]./lam); # RH
end

function get_propagator_mask(field, lam, samp)
   sz = size(field)[1:2]
   sc = 2 .*samp[1]./lam
   return ft2d(jinc.(sqrt.(rr2_sep(real_arr_type(typeof(field)), sz, scale = sc))));
end

function filter_field(field, settings::ImageSettings)
   FD = ft2d(field)
   filtered_field = ift2d(FD.* get_propagator_mask(opt_collect(field), settings.lam, settings.samp));
   return filtered_field
end

function read_translation_matrix(filepath, HoloParams)
   trans_mat = read_hdf5(filepath)
   if HoloParams.pad_fac > 1
      trans_mat .= trans_mat.*HoloParams.pad_fac
   end
   return trans_mat
end

"""
   correct_phase_offset(field, phase_ref)

Corrects the global phase of a complex field to align it with a specified global reference phase.
The function computes the original phase of the summed field and determines the
necessary phase offset. Each element's phase in the field is adjusted accordingly,
resulting in a corrected field that maintains the original magnitudes but has no phase offset to the reference phase.
Phases are wrapped within the range of [-π, π].

# Arguments
+ field: A complex array representing the input field to be corrected
+ phase_ref: The global reference phase to which the field should be aligned
"""
function correct_phase_offset(field, phase_offset)
   field_corr = field.*exp(-1im*phase_offset);
   return field_corr
end

"""
   correct_phase_offset_rh(field, field_ref)

calculates a cross-correlation between a field and a reference field
to determine and correct the relative phase.
"""
function correct_phase_offset_rh(field, field_ref, mask=1)
   phase_offset = angle(sum(field .*conj(field_ref) .* mask))
   field_corr = field.*exp(-1im*phase_offset);
   return field_corr, phase_offset
end

function correct_phase_offset_3D!(field, field_ref, mask=1)
   for idx in range(1, size(field)[3])
      phase_offset = angle(sum(field[:,:,idx] .*conj(field_ref[:,:,idx]) .* mask))
      field[:,:,idx] .= field[:,:,idx].*exp(-1im*phase_offset);
   end
end

function get_pca_mask(fields)
   # vectorize data
   sz1 = 1
   for dim in range(1, length(size(fields))-1)
      sz1 = sz1*size(fields)[dim]
   end
   field_vectorized = reshape(fields, (sz1,size(fields)[end],))
   # seperate imaginary and real parts
   imag_vec = imag.(field_vectorized);
   real_vec = real.(field_vectorized);
   # fit real and imaginary vectorised data with a single value decomposition
   M_imag = fit(PCA, imag_vec; maxoutdim=1, method=:svd); #method=:cov , mean=zeros(size(field_vectorized))
   M_real = fit(PCA, real_vec; maxoutdim=1, method=:svd); #method=:cov, , mean=zeros(size(field_vectorized))
   # project data to principle axis
   proj_imag = projection(M_imag)
   proj_real = projection(M_real)
   # put imaginary and real part back together
   proj_comp = proj_real + 1im.*proj_imag
   # undo vectorisation
   img_proj_pca = reshape(proj_comp, size(fields)[1:end-1])
   return img_proj_pca
end

function correct_phase_offset_pca(fields::Array{T, 3}) where {T}
   mask = get_pca_mask(fields)
   fields_corrected = similar(fields)
   phase_offsets = zeros(size(fields)[end])
   for myidx in range(1,size(fields)[end])
      fields_corrected[:,:, myidx], phase_offsets[myidx] = correct_phase_offset_rh(fields[:,:, myidx], mask)
   end
   return fields_corrected, mask
end

function correct_phase_offset_pca(fields::Array{T, 4}) where {T}
   #mask_dark_pixels!(fields)
   mask = get_pca_mask(fields)
   fields_corrected = similar(fields)
   phase_offsets = zeros(size(fields)[end-1:end])
   for depthidx in range(1,size(fields)[end-1])
      for myidx in range(1,size(fields)[end])
         fields_corrected[:,:,depthidx, myidx], phase_offsets[depthidx,myidx] = correct_phase_offset_rh(fields[:,:,depthidx, myidx], mask[:,:, depthidx])
      end
   end
   return sum(fields_corrected, dims=4), mask
end

function calc_phase_difference(phase1, phase2)
   phase_diff = angle.(exp.(1im.*phase1)./exp.(1im.*phase2))
   return phase_diff
end

function mask_dark_pixels!(field; treshhold=nothing)
   tresh = isnothing(treshhold) ? tresh = maximum(abs.(field))/100 : tresh=treshhold
   field[abs.(field) .< tresh] .= 0.0
end

function calc_correlation_coeff(imag, ref_imag)
   corr_coeff = mean(imag .* conj.(ref_imag)) / sqrt(mean(abs2.(imag)) * mean(abs2.(ref_imag)))
   return corr_coeff
end
