using CUDA
using SpeckleReflections
using SyntheticObjects
using View5D
using Statistics

include("gen_objects.jl")
include("file_io.jl")

function lsd(data, sig_digit=2)
   res = std(data)
   return round(res; sigdigits=2)
end

function my_reclaim()
    GC.gc();
    CUDA.reclaim();
    println("garbage collected, reclaimed CUDA mem.")
end

function simulate_and_reconstruct()
   sz = (501,501,51)
   settings = get_settings(lam=0.488, n=1.5151, NA_illu=1.4, NA_det=1.4,  mag=63, ref_plane=1, samp_z = 0.05, illu_mode=SpeckleReflections.Plane) # give nbr. of images to use during reconstruction
   num_photons = 100
   if (true)
      obj = generate_object_3d(type="resolution_dots"; settings.samp, res_dist=0, size_xyz=sz, dz=24);
      name_simu = "pw_res_dot_z24";
   else
      obj = filaments3D(sz) 
      # obj1 = generate_object_3d(type="Discs"; settings.samp, res_dist=5, size_xyz=sz, dz=50);
      # obj .+= obj1;
      name_simu = "test_fiber_simu_$(num_photons)";
   end

   random_phases = false;
   use_cuda = true
   if (use_cuda)
        obj = cu(obj);
   end
   @time data_all, illu_all, phase_offset = do_simulation(obj, settings; Nimag = 1, random_phases=random_phases, reclaim_fct=my_reclaim, num_photons = num_photons); #   34 sec, 4 Gb GPU, 100 imgs, 301x301x50
   GC.gc()
   CUDA.reclaim()

   if (false)
      folder = raw"../../simulations/resolution_sim/";
      save_all(folder, name_simu, settings, data_all, illu_all, phase_offset)
   end

   #------Reconstruction--Loads the saved data ----------

   use_cuda = true
   if (true) # paste if the code before was not run
      folder = raw"../../simulations/plane_discs/";
      name_simu = "plane_discs";
   end

   preprocess_fct = nothing;
   slice_arr = load_data(folder*name_simu*".h5"; range=1:1)[1];
   pup_dia=(116,116);
   #@vt abs.(SpeckleReflections.ft2d(slice_arr))[]
   if (false)  # If you use this less memory should be consumed.
      pad_factor = 2
      holo_params = get_holo_params(slice_arr, pad_factor; pup_dia=pup_dia)
      ref = gen_ref(slice_arr, settings)
      rescut = rec_holo_shift(slice_arr, holo_params, ref)
      preprocess_fct = (dat) -> rec_holo_shift(dat, holo_params, ref)
   end

   settings, data_all, illu_all, phase_offset = load_all(folder, name_simu; preprocess_fct=preprocess_fct);


   if (false) # just to display the data
      @vt data_all[1]
      @vt illu_all[1]
      # @vt ref_all[1]
      @vt ft(data_all[1])
      # check if illumination k-vector corresponds:
      k_rel = get_rel_ref(data_all[1], (240, 59)) # yields (0.3, -0.3)
   end

   if (use_cuda)
      data_all = cu.(data_all);
      illu_all = cu.(illu_all);
   end;
 
   Nz = 51; # how many slices to reconstruct
   settings = get_settings(settings; samp=(settings.samp[1], settings.samp[2],0.028), ref_plane=1)


   @time rec_obj_pw, phase_offsets_found_zeiss = main_reconstruction(data_all, illu_all, settings, Nz; pup_dia=pup_dia,
                                                                  find_phases=false, calib_mode=CalibDivide, calib_sum=true);
                                                              
   
   
   @vt abs.(rec_obj)[225:275,225:275,:] abs.(rec_obj_pw)[225:275,225:275,:]

   save_data(folder, "rec_"*name_simu, [collect(rec_obj_pw)]) 
   save_data(folder, "found_phase_offsets_zeiss_"*name_simu, phase_offsets_found_zeiss) 

   # @time rec_obj, phase_offsets_found_iter2 = main_reconstruction(data_all, illu_all, settings, Nz; 
   #                                                                phase_iterations=2);

   # @time rec_obj, phase_offsets_found_iter4 = main_reconstruction(data_all, illu_all, settings, Nz; 
   #                                                                phase_iterations=4);

   # mask_thresh=0.10
   # @time rec_obj, phase_offsets_found_iter2m = main_reconstruction(data_all, illu_all, settings, Nz; 
   #                                                                phase_iterations=2, mask_thresh=mask_thresh);

   # 45 sec (CPU) 1 (CUDA) seconds for 100 speckle patterns, 51 slices 301x301
   # @vt abs2.(rec_obj)
   # @vt max.(0,resample(obj, size(rec_obj)))

   # # compare phase offsets found to original
   # rel_phase1 = angle.(exp.(1im.*(phase_offsets_found .- phase_offset)))./2pi;
   # rel_phase2 = angle.(exp.(1im.*(phase_offsets_found_iter2 .- phase_offset)))./2pi;
   # rel_phase2m = angle.(exp.(1im.*(phase_offsets_found_iter2m .- phase_offset)))./2pi;
   # rel_phase0= angle.(exp.(1im.*(0 .- phase_offset)))./2pi;
   # # rel_phase4 = angle.(exp.(1im.*(phase_offsets_found_iter4 .- phase_offset)))./2pi
   # std1 = lsd(rel_phase1)
   # std2 = lsd(rel_phase2)
   # std2m = lsd(rel_phase2m)
   # println("Std.Dev. 1 Iter: $(std1)")
   # println("Std.Dev. 2 Iter: $(std2)")
   # println("Std.Dev. Mask $(mask_thresh*100): $(std2m)")

   # plt = plot(rel_phase1 .- mean(rel_phase1), label="1 iteration: Δ=$(std1)", xlabel="speckle no.", ylabel="relative phase accuracy")
   # plot!(rel_phase2 .- mean(rel_phase2), label="2 iterations: Δ=$(std2)")
   # # plot!(rel_phase4 .- mean(rel_phase4), label="4 iterations")
   # plot!(rel_phase2m .- mean(rel_phase2m), label="mask $(mask_thresh*100)%: Δ=$(std2m)")
   # savefig(plt,folder*name_simu*"_phase_err.png")
   

   # mylene
   folder = folder = raw"../../simulations/test/resolution/";
   rec_obj_div = load_data(folder*"speckle_wave/rec_div_test_resolution_simu.h5"; T=ComplexF64)[1]
   rec_obj = load_data(folder*"speckle_wave/rec_conj_test_resolution_simu.h5"; T=ComplexF64)[1]
   rec_obj_pw = load_data(folder*"plane_wave/rec_test_resolution_plane_simu.h5"; T=ComplexF64)[1]

   psf_div = get_psf(collect(rec_obj_div[:,:,24]), 10;pad_fac=10)
   x = Array(0:length(psf_div)-1) .*new_sampling[1]./10
   plot(x,psf_div, label="division")
   psf_conj = get_psf(collect(rec_obj[:,:,24]), 10;pad_fac=10)
   plot!(x, psf_conj, label="conj", xlabel="x [um]", ylabel="norm. Intensity [a.u.]", title="Lateral Resolution")
   psf_pw = get_psf(collect(rec_obj_pw[:,:,24]), 10;pad_fac=10)
   plot!(x, psf_pw, label="plane wave", xlabel="x [um]", ylabel="norm. Intensity [a.u.]", title="Lateral Resolution")
   savefig(folder*"lateral_resolution.png")

   psf_divz = get_psf(collect(rec_obj_div[199,:,:]), 10;pad_fac=10, normalize=true)
   x = Array(0:length(psf_divz)-1) .*new_sampling[3]./10
   plot(x,psf_divz, label="division")
   psf_conjz = get_psf(collect(rec_obj[199,:,:]), 10;pad_fac=10,normalize=true)
   plot!(x, psf_conjz, label="conj", xlabel="z [um]", ylabel="norm. Intensity [a.u.]", title="Axial Resolution")
   psf_pw = get_psf(collect(rec_obj_pw[199,:,:]), 10;pad_fac=10,normalize=true)
   plot!(x, psf_conjz, label="plane wave", xlabel="z [um]", ylabel="norm. Intensity [a.u.]", title="Axial Resolution")
   savefig(folder*"axial_resolution.png")
end

function get_psf(img, d; psf_loc=nothing, pad_fac=10, normalize=true)
   sz = size(img)
   psf_loc = isnothing(psf_loc) ? [Int(sz[1]/2),Int(sz[2]/2)] : psf_loc
   slice = img[psf_loc[1]-d:psf_loc[1]+d,psf_loc[2]]
   println(size(slice))
   slice_p = abs2.(ift(select_region(ft(slice),new_size= (size(slice).*pad_fac))))
   if normalize
      mx = maximum(slice_p)
      mn = minimum(slice_p)
      slice_p .= (slice_p.-mn)./mx
   end
   return slice_p
end
