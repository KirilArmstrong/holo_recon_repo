"""
    main_reconstruction(data_all, illu_all, settings, Nz; do_crop=false, crop_size=nothing, 
                             pup_dia=(74,74), find_phases=true, phase_iterations=1, mask_thresh=0.0, verbose=false)

    main reconstruction of a series of speckle holograms. 
# Parameters
+ `data_all`:   Vector of either real space data or pre-extracted holograms
+  `illu_all`:  Vector of either real space data or pre-extracted holograms

"""
function main_reconstruction_holoscopy(data_all, settings, pup_dia, abs_pos; do_prop=false, roi=nothing, preserve_size=false, do_oct=true)
    # define size of result
    pad_fac = 1;
    Nz = 2 # propagation does nott work with Nz = 1!!!
    holo_params = get_holo_params(size(data_all[1])[1:2], pad_fac; pup_dia=pup_dia, preserve_size=preserve_size)
    final_size = (holo_params.preserve_size) ? size(data_all[1])[1:2] : pup_dia .* holo_params.pad_fac # holo_params.pad_size
    if isnothing(roi)
        roi = [1:final_size[1], 1:final_size[2]];
    end

    res_obj = zeros(Float32, final_size[1], final_size[2], Nz);
    res_obj = cu(complex(res_obj));
    res_stack = zeros(Float32, length(roi[1]), length(roi[2]), length(settings.lam));
    res_stack = cu(complex(res_stack))
    println(typeof(res_stack))
    #res_obj = similar(complex(data_all[1]), (final_size[1], final_size[2], Nz))
    #res_stack = similar(complex(data_all[1]), (length(roi[1]), length(roi[2]), length(settings.lam)))
    # loop over wavlength
    for idx in range(1, length(settings.lam))
        # define ref wave
        k_ref = get_rel_ref(size(data_all[1][:,:,idx]), (abs_pos[idx,2], abs_pos[idx,1]))
        ref = gen_ref(data_all[1][:,:,idx], k_ref)
        reconstruct_plane_hologram!(res_obj, data_all[1][:,:,idx], holo_params, settings, Nz, settings.lam[idx]; ref=ref) # result is in pupil plane!
        if do_prop
            res_stack[:,:,idx] .= res_obj[roi[1],roi[2],2];
        else
            res_stack[:,:,idx] .= res_obj[roi[1],roi[2],1];
        end
    end

    if do_oct
        res_pad = select_region(res_stack, (size(res_stack)[1:2]...,size(res_stack)[3]*2))
        return ft2d(ift(res_pad,3))
    else
        return res_stack
    end
end

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

