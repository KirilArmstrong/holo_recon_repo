function get_propagator_mask_3d(field, lam, samp, NA=1.0)
   rr2sep = rr2_sep(real_arr_type(typeof(field)), size(field)[1:2], scale = 2 .*samp[1:2].*NA./lam)
   return ft2d(jinc.(sqrt.(rr2sep)));
end

function propagate_field(field::AbstractArray, settings, NA; Nz = size(field, 3), pad_factor=2)
   osz = size(field);
   nsz=(osz[1].*pad_factor, osz[2].*pad_factor, size(field,3)); # pad only first two dimensions!
   field_pad = select_region(field, new_size=nsz);

   PType = complex_arr_type(typeof(expand_dims(field_pad, Val(3))))
   propagator = propagator_col(PType, (size(field_pad)[1:2]..., Nz), settings.samp, settings.lam/settings.n; ref_idx = settings.ref_plane)
   aperture = get_propagator_mask_3d(field_pad, settings.lam, settings.samp, NA)

   propagated_f = ift2d(ft2d(field_pad) .* propagator .* aperture); 

   # crop back to original size
   propagated_field=select_region(propagated_f; new_size=osz);
   return propagated_field;
end

# function propagate_field_3d_r(field::AbstractArray{T, 3}, lam, samp; pad_factor=2, Nz=1, prop_init=nothing) where {T}
#    osz = size(field);
#    nsz=(osz[1].*pad_factor, osz[2].*pad_factor, osz[3]); # pad only first two dimensions!
#    field_pad = select_region(field, new_size=nsz);
#    # propagated = similar(field, nsz);
#    if isnothing(prop_init)
#       if Nz > 1
#          res = [];
#          for slc_nbr in range(1, Nz)
#             proped = propagate_field_2d(field_pad[:,:,slc_nbr], (slc_nbr-1)*samp[3], lam, samp[1:2]);
#             push!(res, proped);
#          end
#          propagated_f = cat(res..., dims=3);
#       else
#          propagated_f =  propagate_field_2d(field_pad, samp[3], lam, samp[1:2]);
#       end
#    else
#       propagated_f =  propagate_field_2d(field_pad, prop_init, lam, samp[1:2]);
#    end
#    # crop back to original size
#    propagated_field=select_region(propagated_f; new_size=osz);
#    return propagated_field;
# end

# function propagate_field_3d_r(field::AbstractMatrix, lam, samp; pad_factor=2, Nz=1, prop_init=nothing)
#    RT = real(eltype(field))
#    lam = RT(lam)
#    samp = RT.(samp)
#    prop_init = isnothing(prop_init) ? nothing : RT.(prop_init)
#    nsz=size(field).*pad_factor;
#    osz = (size(field)[1], size(field)[2],Nz);
#    field_pad = select_region(field, new_size=nsz);
#    propagated_f = similar(field, (nsz...,Nz));
#    if isnothing(prop_init)
#       if Nz > 1
#          # res = [];
#          for slc_nbr in range(1, Nz)
#             prop_dist = (slc_nbr-1)*samp[3]
#             propagated_f[:, :, slc_nbr] .= propagate_field_2d(field_pad, prop_dist, lam, samp[1:2]);
#             # push!(res, proped);
#          end
#          # propagated_f = cat(res..., dims=3);
#       else
#          prop_dist = samp[3] 
#          propagated_f .=  propagate_field_2d(field_pad, prop_dist, lam, samp[1:2]);
#       end
#    else
#       for (slice, dist) in zip(eachslice(propagated_f, dims=3), prop_init)
#          slice .=  propagate_field_2d(field_pad, dist, lam, samp[1:2]);
#       end
#    end
#    # crop back to original size
#    propagated_field=select_region(propagated_f; new_size=osz);
#    return propagated_field;
# end

# function propagate_field_3d(field::AbstractArray{T, 3}, lam, samp; pad_factor=2, Nz=1) where {T}
#    nsz=(size(field)[1].*pad_factor, size(field)[2].*pad_factor, size(field)[3]); # pad only first two dimensions!
#    osz = size(field);
#    field_pad = select_region(field, new_size=nsz);
#    propagated = similar(field, (nsz));
#    prop2d = get_propagator(field_pad, samp[3], lam, samp[1:2]);  # propagation between layers (dz(n) = n*samp[3]; n=1:N)
#    prop3d = prop2d.^reorient(0:Nz-1, Val(3)) .* get_propagator_mask(field_pad, lam, samp) #.* prop2d_init .* get_propagator_mask_3d(field_pad, lam, samp)  .*
#    # propagate
#    if Nz > 1  
#       propagated .= ft2d(field_pad).* prop3d;
#    else
#       if prop_init==0
#          propagated = ft2d(field_pad).* prop2d .* get_propagator_mask(field_pad, lam, samp);
#       else
#          propagated = ft2d(field_pad).* get_propagator(field_pad[1:2], prop_init, lam, samp[1:2]) .* get_propagator_mask(field_pad, lam, samp);
#       end
#    end
#    prop_corr = ift2d(propagated);
   
#    # crop back to original size
#    propagated_field=select_region(prop_corr; new_size=osz);
#    return propagated_field;
# end

# function propagate_field_3d(field::AbstractMatrix, lam, samp; pad_factor=2, Nz=1, prop_init = 0)
#    nsz=size(field).*pad_factor;
#    osz = (size(field)[1], size(field)[2],Nz);
#    field_pad = select_region(field, new_size=nsz);
#    propagated = similar(field, (nsz...,Nz));
#    prop2d = get_propagator(field_pad, samp[3], lam, samp[1:2]);  # propagation between layers (dz(n) = n*samp[3]; n=1:N)
#    prop3d = prop2d.^reorient(0:Nz-1, Val(3)) .* get_propagator_mask(field_pad, lam, samp)
#    # propagate
#    if Nz > 1  
#       propagated .= ft2d(field_pad).* prop3d;
#    else
#       if prop_init==0
#          propagated = ft2d(field_pad).* prop2d.* get_propagator_mask(field_pad, lam, samp);
#       else
#          propagated = ft2d(field_pad).* get_propagator(field_pad, prop_init, lam, samp[1:2]) .* get_propagator_mask(field_pad, lam, samp);
#       end
#    end
#    prop_corr = ift2d(propagated);

   
#    # crop back to original size
#    propagated_field=select_region(prop_corr; new_size=osz);
#    return propagated_field;
# end

# old version
# function propagate_field_3d_old(field, lam, samp; pad_factor=2, Nz=1, distance=0)
#    # To Do: make two functions depending if ndim(field) is 2 or 3
#    if ndims(field) < 3
#       nsz=size(field).*pad_factor;
#       osz = (size(field)[1], size(field)[2],Nz);
#       propagated = similar(field, (nsz...,Nz));
#    else
#       nsz=(size(field)[1].*pad_factor, size(field)[2].*pad_factor, size(field)[3]); # pad only first two dimensions!
#       osz = size(field);
#       propagated = similar(field, (nsz));
#    end
#    field_pad = select_region(field, new_size=nsz);
#    prop2d_init = get_propagator(field_pad, distance, lam, samp[1:2]); # propagation to detector
#    prop2d = get_propagator(field_pad, samp[3], lam, samp[1:2]);  # propagation between layers (dz(n) = n*samp[3]; n=1:N)
#    prop3d = prop2d.^reorient(0:Nz-1, Val(3)) .* get_propagator_mask_3d(field_pad, lam, samp) #.* prop2d_init .* get_propagator_mask_3d(field_pad, lam, samp)  .*
#    # propagate
#    if Nz > 1  
#       propagated .= ft2d(field_pad).* prop3d;
#    else
#         propagated = ft2d(field_pad).* prop2d;
#    end
#    propagated_f = ift2d(propagated);
   
#    # crop back to original size
#    propagated_field=select_region(propagated_f; new_size=osz);
#    return propagated_field;
# end

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
function gen_illumination(obj, settings)
   return gen_illumination(settings.illu_mode, obj, settings);
end

function gen_illumination(::Type{Plane}, obj, settings)
   println("producing plane illumination")
   # plane = ones(ComplexF64,size(obj)[1:2])
   plane = similar(obj, size(obj)[1:2]) # RH
   plane .= 1; # RH
   amp_prop = propagate_field(plane, settings, settings.NA_illu; Nz=size(obj)[3], pad_factor = 1);
   #amp_prop = exp_ikx_col(Array{eltype(obj)}, size(obj), shift_by=(0.0, 0.0, -size(obj,3)*samp[3]/lam)); # 
   return amp_prop
end

function gen_illumination(::Type{Speckle}, obj, settings)
   sz2d = size(obj)[1:2];
   phi_spek = similar(obj, sz2d); # RH
   rand!(phi_spek); # RH
   phi_spek .*= 2pi;
   phi_spek .-= pi;           # make random NxN matrix with range [0, 2pi]
   # rr2sep = rr2_sep(real_arr_type(typeof(obj)), sz2d, scale = 1 ./r_pupil);
   # pup = ft(exp.(1im.*phi_spek)) .*(rr2sep .< 1);
   illu = exp.(1im.*phi_spek) # ift(pup);
   illu = propagate_field(illu, settings, settings.NA_illu; Nz=size(obj)[3], pad_factor = 1);
   return illu;
end

"""
   gen_ref(obj, settings)

generate a reference wave,
#Arguments
+ obj: object for which to generate the reference wave
+ settings: contains the necessary information to calculate the reference wave
"""
function gen_ref(obj, settings)
   RT = real(eltype(obj))
   Nx = size(obj)[1];
   Ny = size(obj)[2];
   # fna = 3 .*[-settings.NA_det, settings.NA_det]; # tilt diagonal -> into settings?
   # k_ref = fna[1].*settings.samp[1].*settings.n./settings.lam
   x = similar(obj, Nx)
   x .= range(0,Nx-1,Nx); # x-coordinte vector
   y = similar(obj, 1, Ny)
   y .= reorient(range(0,Ny-1,Ny),Val(2)); # y-coordinate vector
   line_x = exp.(-x.*(1im.*RT(2pi*settings.k_ref[1]))); # single horizontal line in x of ref field
   line_y = exp.(-y.*(1im.*RT(2pi*settings.k_ref[2]))); # single horizontal line in x of ref field
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
