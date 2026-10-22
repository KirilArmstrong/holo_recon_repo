function custom_hanning(M::Int)
    return [0.5 * (1 - cos(2 * pi * n / (M - 1))) for n in 0:(M-1)]
end

function generate_appodisation_mask(mask_size)
   mask_2d = ones(mask_size[1:2])
   window = custom_hanning(mask_size[3])
   mask = []
   for w in window
      if isempty(mask)
         mask = [mask_2d .* w];
      else
         append!(mask, [mask_2d .* w]);
      end
   end
   return mask
end

function my_reclaim()
    GC.gc();
    CUDA.reclaim();
    println("garbage collected, reclaimed CUDA mem.")
end

"""
    main_reconstruction(data_all, illu_all, settings, Nz; do_crop=false, crop_size=nothing, 
                             pup_dia=(74,74), find_phases=true, phase_iterations=1, mask_thresh=0.0, verbose=false)

    main reconstruction of a series of speckle holograms. 
# Parameters
+ `data_all`:   Vector of either real space data or pre-extracted holograms
+  `illu_all`:  Vector of either real space data or pre-extracted holograms

"""

function main_reconstruction_holoscopy(data_all, settings, pup_dia, abs_pos, Nz; illu_all=nothing, abs_pos_illu=nothing, roi=nothing, preserve_size=false, calib_mode=CalibDivide)
    pad_fac = 1;
    holo_params = get_holo_params(size(data_all)[1:2], pad_fac; pup_dia=pup_dia, preserve_size=preserve_size)
    final_size = (holo_params.preserve_size) ? size(data_all)[1:2] : pup_dia .* holo_params.pad_fac # holo_params.pad_size
    if isnothing(roi)
        roi = [1:final_size[1], 1:final_size[2]];
    end
    # reserve Memory for results
    res_obj = zeros(Float32, final_size[1], final_size[2], Nz);
    res_obj = cu(complex(res_obj));
    res_illu = similar(res_obj);
    res = complex(zeros(ComplexF64, final_size[1], final_size[2], Nz)); #needs to be initialised with zeros otherwise initial values of array will ifluence the result!!
    if calib_mode == CalibConj
        ave_illu = similar(res);
        ave_illu .= 0.0;
    end

    # generate apodisation mask over k
    apod_mask = generate_appodisation_mask((size(res_obj)[1:2]...,length(settings.lam)))
    # loop over k
    for idx in range(1, length(settings.lam))
        println("rec. image nr.", idx," of ", length(settings.lam), " images")
        k_ref = get_rel_ref(size(data_all[:,:,idx]), (abs_pos[idx,1], abs_pos[idx,2]))
        ref = gen_ref(data_all[:,:,idx], k_ref)
        if isnothing(abs_pos_illu)
            ref_illu = ref;
        else
            k_ref = get_rel_ref(size(illu_all[:,:,idx]), (abs_pos_illu[idx,1], abs_pos_illu[idx,2]))
            ref_illu = gen_ref(illu_all[:,:,idx], k_ref);
        end

        # no specific reconstruction for speckle holograms needed anymore!  
        # returns reconstructed field in spatial coordinates  
        reconstruct_plane_hologram!(res_obj, data_all[:,:,idx], holo_params, settings, Nz, settings.lam[idx]; ref=ref)

        settings_illu = get_settings(settings; samp=(settings.samp[1:2]..., -settings.samp[3]))
        if isnothing(illu_all)
            # for plane wave reconstruction, the illumination is simply amp = 1 propagated to the same z as the obje field
            res_illu = gen_illumination(res_obj, settings.lam[idx], settings_illu) # conjugate to invert the Propagation
        else
            reconstruct_plane_hologram!(res_illu, illu_all[:,:,idx], holo_params, settings_illu, Nz, settings_illu.lam[idx]; ref=ref_illu)
        end
        # calibration of fields in spatial coordinates
        obj = calibrate_hologram(collect(res_obj); illu_holo=collect(res_illu), calib_mode=calib_mode);
        # sum over k with appodisation in spatial coordinates (?)
        res += obj.* apod_mask[idx]
        if calib_mode == CalibConj
            ave_illu += calibrate_hologram(collect(res_illu); illu_holo=collect(res_illu), calib_mode=calib_mode);
        end
    end
    # if calib_mode == CalibConj
    #     myeps = sum(abs2.(ave_illu))/prod(size(ave_illu)) ./ 100;
    #     res .*= ave_illu ./ (abs2.(ave_illu) .+ myeps)
    # end
    return res
end

function main_reconstruction_SI_holoscopy(data_all, settings, pup_dia, abs_pos, Nz; illu_all=nothing, abs_pos_illu=nothing, roi=nothing, preserve_size=false, calib_mode=CalibDivide, shifts2=nothing)
    # data_all has shape [m][x,y,k], where m is the index over the speckle positions and k the index over the wavlengths
    pad_fac = 1;
    holo_params = get_holo_params(size(data_all[1])[1:2], pad_fac; pup_dia=pup_dia, preserve_size=preserve_size)
    final_size = (holo_params.preserve_size) ? size(data_all[1])[1:2] : pup_dia .* holo_params.pad_fac # holo_params.pad_size
    if isnothing(roi)
        roi = [1:final_size[1], 1:final_size[2]];
    end
    # reserve Memory for results
    res_obj = zeros(Float32, final_size[1], final_size[2], Nz);
    res_obj = cu(complex(res_obj));
    res_illu = similar(res_obj);
    res = complex(zeros(ComplexF64, final_size[1], final_size[2], Nz)); #needs to be initialised with zeros otherwise initial values of array will ifluence the result!!
    if calib_mode == CalibConj
        ave_illu = similar(res);
        ave_illu .= 0.0;
    end
    data_pr = similar.(data_all, Ref((final_size[1], final_size[2])));
    data_pr = complex.(data_pr);
    data_illu_pr = similar(data_pr);

    # generate apodisation mask over k
    apod_mask = generate_appodisation_mask((size(res_obj)[1:2]...,length(settings.lam)))

    # define function
    function preprocess_fct(dat, myshift, ref)
        return rec_holo_shift(dat, holo_params, ref, myshift)
    end
    # loop over k
    for idx in range(1, length(settings.lam))
        println("rec. image nr.", idx," of ", length(settings.lam), " images")

        # define reference waves
        k_ref = get_rel_ref(size(data_all[1][:,:,idx]), (abs_pos[idx,1], abs_pos[idx,2]))
        ref = gen_ref(data_all[1][:,:,idx], k_ref)
        if isnothing(abs_pos_illu)
            ref_illu = ref;
        else
            k_ref = get_rel_ref(size(illu_all[1][:,:,idx]), (abs_pos_illu[idx,1], abs_pos_illu[idx,2]))
            ref_illu = gen_ref(illu_all[1][:,:,idx], k_ref);
        end

        # create new daat array, whith only the current wavelengths preprocessed hologramms
        for midx in range(1, length(data_all))
            data_pr[midx] = preprocess_fct(data_all[midx][:,:,idx], nothing, ref);
            data_illu_pr[midx] = preprocess_fct(illu_all[1][:,:,idx], nothing, ref_illu);
        end
        datac = cu.(data_pr);
        illuc = cu.(data_illu_pr);
        new_sampling = ((settings.samp[1:2] .* size(data_all[1][:,:,idx]) ./ size(datac[1])[1:2])..., settings.samp[3])
        # define new settings with new sampling
        # ref_plane defines the index in the resulting stack where the propagation is 0
        settings_new = get_settings(settings, samp=new_sampling)
        res_obj, res_illu = reconstruction_si_hologramms(datac, illuc, settings_new, Nz, settings.lam[idx]; find_phases=false, phase_method = "ABS",
                                                            phase_iterations=1, gc_period=5, reclaim_fct=my_reclaim,
                                                            pup_dia=pup_dia, calib_mode=CalibConj, calib_sum=true,
                                                            holo_params=nothing);

        # sum over k with appodisation in spatial coordinates (?)
        res += collect(res_obj); #.* apod_mask[idx]
        ave_illu += collect(res_illu);
    end
    return res, ave_illu
end

# copy pasta from Demonstrator 1 reconstruction...only for preprocessed data!!
function reconstruction_si_hologramms(data_all, illu_all, settings, Nz, lam;   
                             pup_dia=size(data_all[1])[1:2], find_phases=true, phase_method="ABS", phase_iterations=1, 
                             mask_thresh=0.0, verbose=false, calib_mode::Type=CalibConj, calib_sum=true,
                            gc_period = 5, reclaim_fct = GC.gc, holo_params=nothing)
    # define parameters for reconstruction
    pad_factor = 1
    holo_params = isnothing(holo_params) ? get_holo_params(data_all[1], pad_factor; pup_dia=pup_dia, calib_mode=calib_mode) : holo_params;
    
    # the last dimension below will be summed over but this is to check for each depth    
    phase_offsets = zeros(length(data_all))

    pup_dia = isnothing(pup_dia) ? size(data_all[1]) : pup_dia
    final_size = (holo_params.preserve_size) ? size(data_all[1]) : pup_dia .* holo_params.pad_fac # holo_params.pad_size

    ave_illu_zstack_no_ofset = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))
    if phase_method == "PCA"
        rec_img_stack = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz, length(data_all)))
    else
        ave_img_zstack_no_ofset = similar(data_all[1], (final_size[1], final_size[2], Nz))
    end

    res_obj = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))
    res_illu = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))

    rec_field_illu_ft = nothing

    # read in reference wave
    # ref = gen_ref(data_all[1], settings)

    if (!find_phases)
        phase_iterations = 1
    end

    if phase_method == "ABS"
        ref_recon = ave_img_zstack_no_ofset
    end
    for phase_iter = 1:phase_iterations
        if phase_method == "ABS"
            ave_img_zstack_no_ofset .= 0
        end
        ave_illu_zstack_no_ofset .= 0
        for myidx in range(1, length(data_all)) # loops over the speckle patterns
            if (verbose)
                println("processing speckle $(myidx)/$(length(data_all)): ")
            end
            data =  data_all[myidx] # cu(h5read(file_sampel, string(myidx))) : h5read(file_sampel, string(myidx));            

            # read in illumination & reconstruct

            # This is the measured data:
            data_illu = illu_all[myidx] 

            # crop illumination to region of interest
            rec_field_illu_ft = (eltype(data_illu) <: Real) ? rec_holo_shift(data_illu, holo_params, ref) : data_illu

            if settings.illu_mode == Plane
                reconstruct_plane_hologram!(res_obj, data, holo_params, settings, Nz; translation_px=nothing)
                return res_obj, 0.0,phase_offsets
            else
                reconstruct_speckle_hologram!(res_obj, res_illu, data, holo_params, settings, Nz, lam, rec_field_illu_ft; translation_px=nothing)
            end

            if (myidx > 1 && find_phases)
                if phase_method == "ABS"
                    mask = 1
                    if (mask_thresh > 0)
                        ar = abs.(ref_recon)
                        mask = ar .> (maximum(ar) * mask_thresh)
                    end
                    res_corr, phase_offset = correct_phase_offset_rh(res_obj, ref_recon, mask)
                    phase_offsets[myidx] = phase_offset
                elseif phase_method == "PCA"
                    rec_img_stack[:,:,:,myidx] = collect(res_obj)
                end
            else
                res_corr = res_obj;
            end
            # reconstruct image
            if phase_method == "ABS"
                ave_img_zstack_no_ofset .+= res_corr
            end
            ave_illu_zstack_no_ofset .+= res_illu

            if (mod(myidx, gc_period) == 0)
                reclaim_fct();
            end
        end   

        if ((phase_iter < phase_iterations) && phase_method=="ABS")
            ref_recon .= copy(ave_img_zstack_no_ofset)
        end
    end

    if phase_method=="PCA"
        println("correct phase offsets with 3D PCA...")
        ave_img_zstack_no_ofset, pca_mask  = correct_phase_offset_pca(collect(rec_img_stack))
    else
        pca_mask = nothing
    end

    if (calib_sum) # (holo_params.calib_mode != CalibDivide
        myeps = sum(abs2.(ave_illu_zstack_no_ofset))/prod(size(ave_illu_zstack_no_ofset));
        ave_img_zstack_no_ofset .*= ave_illu_zstack_no_ofset ./ (abs2.(ave_illu_zstack_no_ofset) .+ myeps)
        ave_illu_zstack_no_ofset .*= ave_illu_zstack_no_ofset ./ (abs2.(ave_illu_zstack_no_ofset) .+ myeps)
    end
    #println(collect(ave_img_zstack_no_ofset)[1,1,1])
    return ave_img_zstack_no_ofset, ave_illu_zstack_no_ofset
end

function fft_reconstruction_holoscopy(data_all, settings, pup_dia, abs_pos; roi=nothing, preserve_size=false, pad_size=pup_dia)
    pad_fac = 1;
    Nz = size(data_all)[3];
    holo_params = get_holo_params(size(data_all)[1:2], pad_fac; pup_dia=pup_dia, preserve_size=preserve_size)
    final_size = (holo_params.preserve_size) ? size(data_all)[1:2] : pup_dia .* holo_params.pad_fac # holo_params.pad_size
    if isnothing(roi)
        roi = [1:final_size[1], 1:final_size[2]];
    end
    # reserve Memory for results
    res_obj = zeros(Float32, final_size[1], final_size[2], 2);
    res_obj = cu(complex(res_obj));
    res = complex(zeros(ComplexF64, final_size[1], final_size[2], Nz)); #needs to be initialised with zeros otherwise initial values of array will ifluence the result!!
    # generate apodisation mask over k
    # apod_mask = generate_appodisation_mask((size(res_obj)[1:2]...,length(settings.lam)))
    # loop over k
    for idx in range(1, length(settings.lam))
        println("rec. image nr.", idx," of ", length(settings.lam), " images")
        k_ref = get_rel_ref(size(data_all[:,:,idx]), (abs_pos[idx,1], abs_pos[idx,2]))
        ref = gen_ref(data_all[:,:,idx], k_ref)

        # no specific reconstruction for speckle holograms needed anymore!  
        # returns reconstructed field in spatial coordinates  
        reconstruct_plane_hologram!(res_obj, data_all[:,:,idx], holo_params, settings, 2, settings.lam[idx]; ref=ref)

        # sum over k with appodisation in spatial coordinates (?)
        res[:,:,idx]= collect(res_obj[:,:,1]);
    end
    FD = select_region(ft2d(res), (pad_size..., size(res)[3]));
    println(size(FD))
    OCT = ift(FD, 3);
    return ift2d(OCT)
end

# function main_reconstruction_holoscopy(data_all, illu_all=nothing, settings, pup_dia, abs_pos; do_prop=false, roi=nothing, preserve_size=false, do_oct=true)
#     pad_fac = 1;
#     holo_params = get_holo_params(size(data_all[1])[1:2], pad_fac; pup_dia=pup_dia, preserve_size=preserve_size)
#     final_size = (holo_params.preserve_size) ? size(data_all[1])[1:2] : pup_dia .* holo_params.pad_fac # holo_params.pad_size
#     if isnothing(roi)
#         roi = [1:final_size[1], 1:final_size[2]];
#     end

#     res_obj = zeros(Float32, final_size[1], final_size[2], Nz);
#     res_obj = cu(complex(res_obj));
#     res_illu = simmilar(res_obj);
#     res = simmilar(res_obj);
#     Nz = 100;
#     # loop over k
#     for idx in range(1, length(settings.lam))
#         k_ref = get_rel_ref(size(data_all[1][:,:,idx]), (abs_pos[idx,2], abs_pos[idx,1]))
#         ref = gen_ref(data_all[1][:,:,idx], k_ref)
#         reconstruct_plane_hologram!(res_obj, data_all[1][:,:,idx], holo_params, settings, Nz, settings.lam[idx]; ref=ref) # result is in pupil plane!
#         if isnothing(illu_all)
#             # for plane wave reconstruction, the illumination is simply amp = 1 propagated to the same z as the obje field
#             new_sampling = ((settings.samp[1:2] .* size(data_all)[1:2] ./ size(res_obj)[1:2])..., settings.samp[3])
#             settings_illu = get_settings(settings; illu_mode=Plane, sampling=new_sampling)
#             res_illu = gen_illumination(res_obj, settings.lam[idx], settings_illu)
#         else
#             reconstruct_plane_hologram!(res_illu, illu_all[1][:,:,idx], holo_params, settings, Nz, settings.lam[idx]; ref=ref) # result is in pupil plane!
#         end
#         res += res_obj ./ res_illu # sum over k
#     end
#     return res
# end

# function main_reconstruction_holoscopy(data_all, settings, pup_dia, abs_pos; do_prop=false, roi=nothing, preserve_size=false, do_oct=true)
#     # define size of result
#     pad_fac = 1;
#     Nz = 2 # propagation does nott work with Nz = 1!!!
#     holo_params = get_holo_params(size(data_all[1])[1:2], pad_fac; pup_dia=pup_dia, preserve_size=preserve_size)
#     final_size = (holo_params.preserve_size) ? size(data_all[1])[1:2] : pup_dia .* holo_params.pad_fac # holo_params.pad_size
#     if isnothing(roi)
#         roi = [1:final_size[1], 1:final_size[2]];
#     end

#     res_obj = zeros(Float32, final_size[1], final_size[2], Nz);
#     res_obj = cu(complex(res_obj));
#     res_stack = zeros(Float32, length(roi[1]), length(roi[2]), length(settings.lam));
#     res_stack = cu(complex(res_stack))
#     println(typeof(res_stack))
#     #res_obj = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))
#     #res_stack = similar(complex(data_all[1]), (length(roi[1]), length(roi[2]), length(settings.lam)))
#     # loop over wavlength
#     for idx in range(1, length(settings.lam))
#         # define ref wave
#         k_ref = get_rel_ref(size(data_all[1][:,:,idx]), (abs_pos[idx,2], abs_pos[idx,1]))
#         ref = gen_ref(data_all[1][:,:,idx], k_ref)
#         reconstruct_plane_hologram!(res_obj, data_all[1][:,:,idx], holo_params, settings, Nz, settings.lam[idx]; ref=ref) # result is in pupil plane!
#         if do_prop
#             res_stack[:,:,idx] .= res_obj[roi[1],roi[2],2];
#         else
#             res_stack[:,:,idx] .= res_obj[roi[1],roi[2],1];
#         end
#     end

#     if do_oct
#         res_pad = select_region(res_stack, (size(res_stack)[1:2]...,size(res_stack)[3]*2))
#         return ft2d(ift(res_pad,3))
#     else
#         return res_stack
#     end
# end

#     return ift(res_stack,3)
# end

# function main_reconstruction(data_all, illu_all, settings, Nz, lam;   
#                              pup_dia=size(data_all[1])[1:2], find_phases=true, phase_method="ABS", phase_iterations=1, 
#                              mask_thresh=0.0, verbose=false, calib_mode::Type=CalibConj, calib_sum=true,
#                             gc_period = 5, reclaim_fct = GC.gc, holo_params=nothing)
#     # define parameters for reconstruction
#     # crop_params = (do_crop) ? CropParams(crop_size,  (size(data_all[1])[1:2] .÷2 .+1)) : CropParams(size(data_all[1]), size(data_all[1])[1:2] .÷2 .+1)
#     pad_factor = 2
#     holo_params = isnothing(holo_params) ? get_holo_params(data_all[1], pad_factor; pup_dia=pup_dia, calib_mode=calib_mode) : holo_params;
    
#     # the last dimension below will be summed over but this is to check for each depth    
#     phase_offsets = zeros(length(data_all))

#     pup_dia = isnothing(pup_dia) ? size(data_all[1]) : pup_dia
#     final_size = (holo_params.preserve_size) ? size(data_all[1]) : pup_dia .* holo_params.pad_fac # holo_params.pad_size

#     ave_illu_zstack_no_ofset = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))
#     if phase_method == "PCA"
#         rec_img_stack = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz, length(data_all)))
#     else
#         ave_img_zstack_no_ofset = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))
#     end

#     res_obj = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))
#     res_illu = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))

#     rec_field_illu_ft = nothing

#     # read in reference wave
#     ref = gen_ref(data_all[1], settings)

#     if (!find_phases)
#         phase_iterations = 1
#     end

#     if phase_method == "ABS"
#         ref_recon = ave_img_zstack_no_ofset
#     end
#     for phase_iter = 1:phase_iterations
#         if phase_method == "ABS"
#             ave_img_zstack_no_ofset .= 0
#         end
#         ave_illu_zstack_no_ofset .= 0
#         for myidx in range(1, length(data_all)) # loops over the speckle patterns
#             if (verbose)
#                 println("processing speckle $(myidx)/$(length(data_all)): ")
#             end
#             data =  data_all[myidx] # cu(h5read(file_sampel, string(myidx))) : h5read(file_sampel, string(myidx));            

#             # read in illumination & reconstruct

#             # This is the measured data:
#             data_illu = illu_all[myidx] 

#             # crop illumination to region of interest
#             rec_field_illu_ft = (eltype(data_illu) <: Real) ? rec_holo_shift(data_illu, holo_params, ref) : data_illu

#             if settings.illu_mode == Plane
#                 reconstruct_plane_hologram!(res_obj, data, holo_params, settings, Nz, lam; ref=ref)
#                 return res_obj, 0.0,phase_offsets
#             else
#                 reconstruct_speckle_hologram!(res_obj, res_illu, data, holo_params, settings, Nz, rec_field_illu_ft; ref=ref, translation_px=nothing)
#             end

#             if (myidx > 1 && find_phases)
#                 if phase_method == "ABS"
#                     mask = 1
#                     if (mask_thresh > 0)
#                         ar = abs.(ref_recon)
#                         mask = ar .> (maximum(ar) * mask_thresh)
#                     end
#                     res_corr, phase_offset = correct_phase_offset_rh(res_obj, ref_recon, mask)
#                     phase_offsets[myidx] = phase_offset
#                 elseif phase_method == "PCA"
#                     rec_img_stack[:,:,:,myidx] = collect(res_obj)
#                 end
#             else
#                 res_corr = res_obj;
#             end
#             # reconstruct image
#             if phase_method == "ABS"
#                 ave_img_zstack_no_ofset .+= res_corr
#             end
#             ave_illu_zstack_no_ofset .+= res_illu

#             if (mod(myidx, gc_period) == 0)
#                 reclaim_fct();
#             end
#         end   

#         if ((phase_iter < phase_iterations) && phase_method=="ABS")
#             ref_recon .= copy(ave_img_zstack_no_ofset)
#         end
#     end

#     if phase_method=="PCA"
#         println("correct phase offsets with 3D PCA...")
#         ave_img_zstack_no_ofset, pca_mask  = correct_phase_offset_pca(collect(rec_img_stack))
#     else
#         pca_mask = nothing
#     end

#     if (calib_sum) # (holo_params.calib_mode != CalibDivide
#         @show myeps = sum(abs2.(ave_illu_zstack_no_ofset))/prod(size(ave_illu_zstack_no_ofset)) ./ 100;
#         ave_img_zstack_no_ofset .*= ave_illu_zstack_no_ofset ./ (abs2.(ave_illu_zstack_no_ofset) .+ myeps)
#         ave_illu_zstack_no_ofset .*= ave_illu_zstack_no_ofset ./ (abs2.(ave_illu_zstack_no_ofset) .+ myeps)
#     end
#     #println(collect(ave_img_zstack_no_ofset)[1,1,1])
#     return ave_img_zstack_no_ofset, pca_mask, ave_illu_zstack_no_ofset
# end

