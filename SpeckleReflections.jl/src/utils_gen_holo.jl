function get_propagator_mask_3d(field, lam, samp, NA=1.0)
   rr2sep = rr2_sep(real_arr_type(typeof(field)), size(field)[1:2], scale = 2 .*samp[1:2].*NA./lam)
   return ft2d(jinc.(sqrt.(rr2sep)));
end

function propagate_field(field::AbstractArray, lam, settings, NA; Nz = size(field, 3), pad_factor=2)
   osz = size(field);
   nsz=(osz[1].*pad_factor, osz[2].*pad_factor, size(field,3)); # pad only first two dimensions!
   field_pad = select_region(field, new_size=nsz);

   PType = complex_arr_type(typeof(expand_dims(field_pad, Val(3))))
   propagator = propagator_col(PType, (size(field_pad)[1:2]..., Nz), settings.samp, lam/settings.n; ref_idx = 1)
   aperture = get_propagator_mask_3d(field_pad, lam, settings.samp, NA)

   propagated_f = ift2d(ft2d(field_pad) .* propagator .* aperture); 

   # crop back to original size
   propagated_field=select_region(propagated_f; new_size=osz);
   return propagated_field;
end

"""
   gen_illumination(obj, settings)

generate an illumination field of the Form 1*exp(1im*phi(x,y)).
For 2D Objects the illumination will also be 2D. For 3D Objects the
Illumination will be in form of a matrix where each Layer is propagated
to the according layer in the object.
#Arguments
+ Type: Either Speckle or Plane wave Illumination
+ obj: object for which to generate the illumination field
+ lam: wavelength
+ samp: sampling of the object
+ NA: NA of the illumination
"""
function gen_illumination(obj, lam, settings)
   return gen_illumination(settings.illu_mode, obj, lam, settings);
end

function gen_illumination(::Type{Plane}, obj, lam, settings)
   println("producing plane illumination")
   # plane = ones(ComplexF64,size(obj)[1:2])
   plane = similar(obj, size(obj)[1:2]) # RH
   plane .= 1; # RH
   amp_prop = propagate_field(plane, lam, settings, settings.NA_illu; Nz=size(obj)[3], pad_factor = 1);
   #amp_prop = exp_ikx_col(Array{eltype(obj)}, size(obj), shift_by=(0.0, 0.0, -size(obj,3)*samp[3]/lam)); # 
   return amp_prop
end

function gen_illumination(::Type{Speckle}, obj, lam, settings)
   sz2d = size(obj)[1:2];
   phi_spek = similar(obj, sz2d); # RH
   rand!(phi_spek); # RH
   phi_spek .*= 2pi;
   phi_spek .-= pi;           # make random NxN matrix with range [0, 2pi]
   # rr2sep = rr2_sep(real_arr_type(typeof(obj)), sz2d, scale = 1 ./r_pupil);
   # pup = ft(exp.(1im.*phi_spek)) .*(rr2sep .< 1);
   illu = exp.(1im.*phi_spek) # ift(pup);
   illu = propagate_field(illu, lam, settings, settings.NA_illu; Nz=size(obj)[3], pad_factor = 1);
   return illu;
end

"""
   gen_ref(obj, settings)

generate a reference wave,
#Arguments
+ obj: object for which to generate the reference wave
+ settings: contains the necessary information to calculate the reference wave
"""
function gen_ref(obj, k_ref)
   RT = real(eltype(obj))
   Nx = size(obj)[1];
   Ny = size(obj)[2];
   # fna = 3 .*[-settings.NA_det, settings.NA_det]; # tilt diagonal -> into settings?
   # k_ref = fna[1].*settings.samp[1].*settings.n./settings.lam
   x = similar(obj, Nx)
   x .= range(0,Nx-1,Nx); # x-coordinte vector
   y = similar(obj, 1, Ny)
   y .= reorient(range(0,Ny-1,Ny),Val(2)); # y-coordinate vector
   line_x = exp.(-x.*(1im.*RT(2pi*k_ref[1]))); # single horizontal line in x of ref field
   line_y = exp.(-y.*(1im.*RT(2pi*k_ref[2]))); # single horizontal line in x of ref field
   field = line_x .* line_y; # use the outer product to calculate the 2D reference wave fast
   return field;
end

"""
   preserve_center_only!(holo, sz)

sets the outer part of an array (holo) to zero, sz denoting the size of the region to preserve.
sz is required to be odd.
"""
# function preserve_center_only!(holo, sz)
#    mask = disc(size(holo)[1:2], sz)
#    holo .= holo .*mask
#    # midpos = center(size(holo), CenterFT);
#    # radius = (sz.-1).÷2; # 
#    # holo[1:midpos[1]-radius[1]-1,:] .= zero(eltype(holo));
#    # holo[midpos[1]+radius[1]+1:end,:] .= zero(eltype(holo));
#    # holo[:,1:midpos[2]-radius[2]-1] .= zero(eltype(holo));
#    # holo[:,midpos[2]+radius[2]+1:end] .= zero(eltype(holo));
# end

"""
   calc_pixel_shift(FD, fna, samp, lam, pad_fac)

calculates the number of pixels a specific frequency has in fourier space.
The frequency is calculated by the given angle as follows: freq = fna/lam
#Arguments
+ FD: The Fourier Transformed Signal
+ fna: The angle of the reference wave in NA-units, fna = sin(angle)
+ samp: sampling of the object
+ lam: wavelength
+ pad_fac: padding factor used before FT
"""
function calc_pixel_shift(FD, fna, samp, lam, pad_fac)
   # calculate center coordinates of Interference Freq.:
   Nxy = size(FD)[1:2]; # -> if N gets bigger, df gets smaller but f_max stays the same
   dxy = samp[1:2]./pad_fac # ./pad_fac; # account for Nxy = N* pad_fac
   pixel_shift = fna.*dxy.*Nxy./lam; # Pixel shift from center pixel , for padding = 1
   ctr = NDTools.center(Nxy, CenterFT);
   center_coord = pixel_shift.*pad_fac .+ ctr; # multiply pixelshift with pad factor and add center Pixel coordinates (Nxy/2, Nxy/2)
   return center_coord
end

"""
   calc_fna_shift(FD, coord, samp, lam, pad_fac)

Calculates the angle according to a specific pixel shift in Fourier Domain.
In essence does the inverse calculation of calc_pixel_shift()
#Arguments
+ FD: The Fourier Transformed Signal
+ coord: The pixel coordinates in FD where the x and y coordinates are in the range of 0 to N
+ samp: sampling of the object
+ lam: wavelength
+ pad_fac: padding factor used before FT
"""
function calc_fna_shift(FD, coord, samp, lam, pad_fac)
   # calculate the angle according to a specific pixel shift
   # in essence does the inverse calculation of calc_pixel_shift()
   Nxy = size(FD)[1:2]; # -> if N gets bigger, df gets smaller but f_max stays the same
   dxy = samp[1:2]./pad_fac # ./pad_fac; # account for Nxy = N* pad_fac
   ctr = NDTools.center(Nxy, CenterFT);
   px_shift = round.(Int,(coord.-ctr)./pad_fac);
   fna = px_shift.*lam./(dxy.*Nxy);
   return fna
end

# """
#    cexport_data(data, filename)

# Simple Routine to export the given data to a .mat or .npz file
# """
# function export_data(data; filename="data", type=".txt")
#    if type == ".m"
#       file = matopen("test\\"*filename*type, "w")
#       write(file, "data", data)
#       close(file)
#    elseif type==".npz"
#       npzwrite("test\\"*filename*type,data)
#    elseif type == ".txt"
#       file = "test\\"*filename*type
#       open(file, "w") do io
#          for idx in range(1, size(data)[3]) 
#             writedlm(io, data[:,:,idx,1])
#          end
#      end
#    else
#       print("\n Error: unvalid file type. Please use type .m or .npz. given type: ", type)
#    end
# end

function apply_lens(field2D, r_pup_det)
   RT = real(eltype(field2D))
   r_pup_det = RT(r_pup_det)
   obj_L1 = ft(select_region(field2D, new_size=size(field2D).*(2,2,1))) # zeropad in x,y before ft to avoid wraparounds
   # detection_pupil = disc(size(obj_L1)[1:2],(r_pup_det, r_pup_det)); # RH
   detection_pupil = rr2_sep(real_arr_type(typeof(field2D)), size(obj_L1)[1:2], scale=1 ./r_pup_det) .< 1; # RH
   obj_pup = obj_L1 .* detection_pupil;
   obj_L2 = select_region(ift(obj_pup), new_size=size(field2D))
   return obj_L2
end
