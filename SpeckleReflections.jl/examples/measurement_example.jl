using SpeckleReflections
using FindShift # for get_translation_mat
using View5D
using CUDA
using Statistics
include("file_io.jl")

function my_reclaim()
    GC.gc();
    CUDA.reclaim();
    println("garbage collected, reclaimed CUDA mem.")
end

function calculate_theoretical_pixel_shift(settings, dum; N=100, Npxy=10)
   shifts = Vector{Vector{Any}}(undef, N)
   dpx = dum/settings.samp[1] # original pixelsize on sampel = 2.5/settings.mag, 
   println("pixelshift: ", dpx)
   dx = 0:dpx:(Npxy-1)*dpx
   dy = 0:dpx:(Npxy-1)*dpx
   index = 0
   for x in dx
      for y in dy
         index +=1
         shifts[index] = [-x,-y,0]
      end
   end
   return shifts
end

"""
   get_translations(data)

calculates the translation vectors between images by first subtracting the mean image and then
using the `align_stack` routine from the `FindShift.jl` toolbox. 
"""
function get_translations(data, ref; red_dim=3, reg=1.0, max_freq=0.15)
   # stack images
   image_stack = cat(data..., dims=red_dim);
   # ref_stack = cat(ref..., dims=red_dim);

   # # define function to do abs2 elementwise
   # doabs2(x) = abs2.(x)

   # # substract background?
   # image_stack .*= ref_stack ./ (doabs2.(ref_stack) .+ reg)
   # image_stack .-= sum(ref_stack)/length(ref_stack)

   # find shifts and align
   aligned, shifts = align_stack(image_stack, refno=1, damp=0.1, max_freq=max_freq)

   return shifts, aligned;
end

function get_translations_main_rh(data, data_illu, settings; red_dim=3, reg=1.0, pup_dia=nothing)
   if isnothing(pup_dia)
      pup_dia = (570*2,570*2)
   end
   holo_params = get_holo_params(size(data[1]), 2; pup_dia=pup_dia)
   ref = gen_ref(data[1], settings)

   doabs(x) = abs.(x)
   datcut = doabs.(SpeckleReflections.FourierTools.ift.(rec_holo_shift.(data, Ref(holo_params), Ref(ref))));
   illucut = abs.(SpeckleReflections.FourierTools.ift(rec_holo_shift(data_illu[1], holo_params, ref)));
       
   image_stack = cat(datcut..., dims=red_dim)
   # calculate bakground (speckle pattern, etc.)
   # bkg = sum(image_stack, dims=red_dim)./Nimag

   # define 1. image to be the reference to which all images should be shifted
   # image_stack .-= bkg
   image_stack .*= illucut ./ (abs2.(illucut) .+ reg)
   image_stack .-= sum(image_stack)/length(image_stack)
   aligned, shifts = align_stack(image_stack, refno=1, damp=0.1, max_freq=0.1)
   return shifts, aligned;
end

function get_translations_main_ma(file_sample, file_illu, pre_shift_um, settings; abs_pos=nothing, pup_dia=nothing, frac=nothing, roi=nothing, bkg_sub=true, max_freq=0.15)
   
   # read in non cropped data
   data = load_data(file_sample, roi);
   data_illu = similar(data);
   data_illu .= load_data(file_illu, roi);


   # define reconstruction parameters and settings from FFT plot
   if isnothing(pup_dia)
      @vr sqrt.(abs.(SpeckleReflections.ft2d(data[1])))
      abs_pos = (918*2,513*2)
      pup_dia = (570*2,570*2)
   end

   k_ref = get_rel_ref(size(data[1]), abs_pos)
   holo_params = get_holo_params(size(data[1]), 2; pup_dia=pup_dia)
   settings = get_settings_meas(mag=63, samp_z=0.1, k_ref=k_ref, NA_det=1.4, NA_illu=1.4)
   ref = gen_ref(data[1], settings)

   # reconstruct holograms (not croped to ROI)
   datcut = rec_holo_shift.(data, Ref(holo_params), Ref(ref));
   illucut = rec_holo_shift(data_illu[1], holo_params, ref);
   doabs(x) = abs.(x) # function to do abs elementwise
   image_stack = doabs.(SpeckleReflections.FourierTools.ift.(datcut))
   illu_stack = abs.(SpeckleReflections.FourierTools.ift(illucut))

   # do background substraction (only beneficial if it is a mirror measurement)
   if bkg_sub
      bkg = [sum(image_stack)./size(image_stack)]
      image_stack .= image_stack .- bkg
   end

   # define new sampling due to cropping in Fourier Space
   frac = size(data[1])[1:2] ./ size(datcut[1])
   new_sampling = ((settings.samp[1:2] .* frac)..., settings.samp[3])
   settings_new = get_settings(settings, samp=new_sampling)

   # do course backshifting on measurement knowledge (we know how we theoretically shifted)
   pre_shifts = calculate_theoretical_pixel_shift(settings_new, pre_shift_um; N=length(image_stack), Npxy=Int(sqrt(length(image_stack))))
   # image_stack .= SpeckleReflections.shift.(image_stack, pre_shifts);
   # illu_stack_full = SpeckleReflections.shift.(Ref(illu_stack),  pre_shifts);

   # do fine shift based on course shifted images
   nsz = Tuple(round.(Int,size(image_stack[1]) .- abs.(pre_shifts[end][1:2])));
   ncent = nsz.÷2
   image_stack = SpeckleReflections.select_region.(SpeckleReflections.shift.(image_stack, pre_shifts); new_size=nsz, center=ncent);
   illu_stack_full = SpeckleReflections.select_region.(SpeckleReflections.shift.(Ref(illu_stack),  pre_shifts); new_size=nsz, center=ncent);
   fine_shifts, aligned = get_translations(image_stack,illu_stack_full , max_freq=max_freq)
  
  # fine_shifts, aligned = get_translations(image_stack, illu_stack_full, max_freq=max_freq)
   shifts = pre_shifts .+ fine_shifts # these shifts are scaled due to new sampling
   @vt sum(aligned, dims=3) sum(image_stack)

   # backscale shifts so they can be used on raw data
   frac = size(data[1])./size(datcut[1]);
   shifts .= [ashift .* [frac[1], frac[2], 0.0] for ashift in shifts]
   pre_shifts .= [ashift .* [frac[1], frac[2], 0.0] for ashift in pre_shifts]

   # save
   # save_translations(shifts, file_sample[1:end-3])

   return shifts, pre_shifts
end

function plot_shifts(shifts, shifts_theo; fname=nothing)
   x_vec = []
   y_vec = []
   for vec in shifts
      push!(x_vec, vec[1])
      push!(y_vec, vec[2])
   end
   x_theo = []
   y_theo = []
   for vec in shifts_theo
      push!(x_theo, vec[1])
      push!(y_theo, vec[2])
   end
   plt = plot([x_theo y_theo],label=["x theo" "y theo"], xlabel="Image Nr.", ylabel="translation in pixels", title="Extracted translation values")
   plot!([x_vec y_vec],label=["x extracted" "y extracted"], xlabel="Image Nr.", ylabel="translation in pixels", title="Extracted translation values")
   
   return plt
end

function convert_stuff()
      folder = raw"D:\data\Mylene_data\USAF\\"
      file = raw"20260209_illuminatiuon_110us_integration_time"
      convert_hdf5(folder, file)      
      folder = raw"C:\NoBackup\Data\Mylene\USAF\\"
      file = raw"20260209_USAF_150us_integration_time"
      convert_hdf5(folder, file)      
end

function find_axial_shift(dat_illu, data_obj, settings, holo_params, nbr_iter, dz; roi=nothing, prop_init=0.0)
   ref = gen_ref(data_obj[1], settings)
   function preprocess_fct(dat)
      return rec_holo_shift(dat, holo_params, ref)
   end

   function calc_corr_coeff(imag, ref_imag)
      return mean(imag.*conj.(ref_imag)) / sqrt(mean(abs2.(imag)) .* mean(abs2.(ref_imag)));
   end
   
   # preprocess data
   data_illu_pr = preprocess_fct.(dat_illu);
   data_pr = SpeckleReflections.ift2d.(preprocess_fct.(data_obj));
   Nz = nbr_iter
   new_sampling = ((settings.samp[1:2] .* size(data_illu[1])[1:2] ./ size(data_illu_pr[1])[1:2])..., dz)
   settings = get_settings(settings, samp=new_sampling, ref_plane=1)

   # predefine propagation aperture and datatype
   PType = SpeckleReflections.complex_arr_type(typeof(SpeckleReflections.expand_dims(data_illu_pr[1], Val(3))))
   aperture = SpeckleReflections.get_propagator_mask_3d(data_illu_pr[1], settings.lam, new_sampling[1:2], settings.NA_det)
      
   # initial global propagation
   if prop_init==0.0
      start_idx = 2
   else
      start_idx = round(Int, prop_init/dz)+1; # plus 1 because first slice of propagator is always z=0.0
      Nz = start_idx +Nz
   end


   # define propagator for propagation stack
   propagator = SpeckleReflections.propagator_col(PType, (size(data_illu_pr[1])[1:2]..., Nz), new_sampling, settings.lam; ref_idx = settings.ref_plane)
   prop_dist = []
   corr_val_mat = []
   img_nbr = 1
   for data in data_pr
      println("correlating image nr.: ", img_nbr)    
      illu_prop = SpeckleReflections.ift2d(data_illu_pr[img_nbr] .* conj.(propagator[:,:,start_idx:end] .* aperture));
      corr_val = []
      for prop_idx in range(1,size(illu_prop,3))
         coeff = calc_corr_coeff(abs.(illu_prop[roi[1],roi[2],prop_idx]), abs.(data[roi[1], roi[2]]))
         append!(corr_val, coeff)
      end
      max_val, corr_idx = findmax(corr_val)
      append!(corr_val_mat, [corr_val])
      append!(prop_dist, corr_idx*dz + prop_init)
      img_nbr +=1
   end
   return prop_dist, corr_val_mat
end

function reconstruct_measured_data()
   ##--- define files and sizes ----------------------------------------------
   # define folders and filenames
   folder = raw"/home/zomamstu/measurements_temp//"
   file_sample = folder * "image_discrete_pw_steps_100imgs_z3659_1pos.h5"
   file_illu = folder * "image_discrete_pw_mirror_z2453_100img_1pos.h5"
   # file_trans_mat = file_sample[1:end-3] * "_translations.h5"

   ##--- define reagion of interest and get sizes ----------------------------------------------
   data_raw = load_data(file_sample)
   illu_raw = load_data(file_illu)
   ct = size(data_raw[1])[1:2] .÷2; # [2460, 1840]
   rad = [1374, 1000];
   roi = [ct[1]-rad[1]+1:ct[1]+rad[1], ct[2]-rad[2]+1:ct[2]+rad[2]]
   # @vt data_raw[1][roi[1], roi[2],2]

   # ----- get position of 1. order ------------ #
   # @vr sqrt.(abs.(SpeckleReflections.ft2d(data_raw[1][:,:,1];)))
   abs_pos_start = Tuple(round.(Int, (854, 700).*length.(roi)./size(data_raw[1])[1:2]));
   pup_dia = Tuple(round.(Int, (1500, 1100).*length.(roi)./size(data_raw[1])[1:2]));
   # @vr sqrt.(abs.(SpeckleReflections.ft2d(data_raw[1][:,:,end];)))
   abs_pos_end = Tuple(round.(Int, (985, 784).*length.(roi)./size(data_raw[1])[1:2]));
   # @vr sqrt.(abs.(SpeckleReflections.ft2d(data_raw[1][roi[1],roi[2],end])))[abs_pos_end[1]-pup_dia[1]÷2:abs_pos_end[1]+pup_dia[1]÷2,abs_pos_end[2]-pup_dia[2]÷2:abs_pos_end[2]+pup_dia[2]÷2]
   abs_pos_x = LinRange.(abs_pos_start[1], abs_pos_end[1], size(data_raw[1],3));
   abs_pos_y = LinRange.(abs_pos_start[2], abs_pos_end[2], size(data_raw[1],3));
   abs_pos = cat(abs_pos_y, abs_pos_x, dims=2)
   # FD = SpeckleReflections.ft2d(data_raw[1][roi[1],roi[2],end])[abs_pos_end[1]-pup_dia[1]÷2:abs_pos_end[1]+pup_dia[1]÷2,abs_pos_end[2]-pup_dia[2]÷2:abs_pos_end[2]+pup_dia[2]÷2]
   # @vt abs.(FD)

   # @vr sqrt.(abs.(SpeckleReflections.ft2d(illu_raw[1][:,:,1];)))
   # @vr sqrt.(abs.(SpeckleReflections.ft2d(illu_raw[1][:,:,end];)))
   abs_pos_start = Tuple(round.(Int, (842, 715).*length.(roi)./size(data_raw[1])[1:2]));
   abs_pos_end = Tuple(round.(Int, (970, 801).*length.(roi)./size(data_raw[1])[1:2]));
   abs_pos_x = LinRange.(abs_pos_start[1], abs_pos_end[1], size(data_raw[1],3));
   abs_pos_y = LinRange.(abs_pos_start[2], abs_pos_end[2], size(data_raw[1],3));
   abs_pos_illu = cat(abs_pos_y, abs_pos_x, dims=2)
   # FD = SpeckleReflections.ft2d(illu_raw[1][roi[1],roi[2],1])[abs_pos_start[1]-pup_dia[1]÷2:abs_pos_start[1]+pup_dia[1]÷2,abs_pos_start[2]-pup_dia[2]÷2:abs_pos_start[2]+pup_dia[2]÷2]
   # @vt abs.(FD)

   
   # get lamdas
   hdf5obj = h5open(file_sample)
   lam_vec = attrs(hdf5obj["1"])["lambdas"] ./1000;
   # lam_str = attrs(hdf5obj["1"])["lambdas"]
   # lam_vec = parse.(Float32, split(strip(lam_str, ['[', ']'])))./1000;
   #k_vec=2*π./lam_vec;
   settings = get_settings(lam=lam_vec, n=1.0, NA_illu=0.16, NA_det=0.16,  mag=5, samp_z = 1.0, illu_mode=SpeckleReflections.Plane) # give nbr. of images to use during reconstruction

   ##--- reconstruct on GPU using CUDA ----------------------------------------------
   #data_raw = load_data(file_sample, [roi...,:])
   datac = [cu(data_raw[1][roi[1], roi[2], :])];
   illuc = [cu(illu_raw[1][roi[1], roi[2], :])];
   # illuc = cu.(data_illu_pr);
   GC.gc();
   CUDA.reclaim();

   # calculate new sampling due to cropping out first order during preprocessing
   new_sampling = ((settings.samp[1:2] .* length.(roi) ./ size(datac[1])[1:2])..., 7.5)
   # define new settings with new sampling
   # ref_plane defines the index in the resulting stack where the propagation is 0
   settings_new = get_settings(settings, samp=new_sampling, lam=lam_vec)
   final_sz = size(datac[1])[1:2]
   res_roi = [1:final_sz[1], 1300-1:1300]
   
   Nz = 100;
   rec_obj = SpeckleReflections.main_reconstruction_holoscopy(datac[1], settings_new, pup_dia, abs_pos, Nz;illu_all=illuc[1], abs_pos_illu=abs_pos_illu)#

   GC.gc();
   CUDA.reclaim();

   # show result
   @vt abs.(rec_obj)

   # save result
   save_data(folder, "rec_750um_calib_"*"image_discrete_pw_steps_100imgs_z3659_1pos", [collect(rec_obj)],".h5")

   # save single slice as image
   save_obj = ave_img_theo[359:559,624:824]./ave_illu_theo[359:559,624:824]
   @vt save_obj
   save(folder*"incoherent_sum_zoomed_pic1_theo.png",(save_obj.-minimum(save_obj))./(maximum(save_obj.-minimum(save_obj))))



   ##--- find shifts (only needs to be done once) ----------------------------------------------
   # since this is a huge code block it was put in a function.
   # However, the naming was untouched so code can be sent manualy to the REPL from within the function
   # if (false)
   #    # MA approach
   #    shifts2, shifts_theo = get_translations_main_ma(file_sample, file_illu, 1.0, settings; abs_pos=abs_pos, pup_dia=pup_dia, roi=roi, bkg_sub=true, max_freq=1.0)
   #    plt = plot_shifts(shifts2, shifts_theo)
      
   #    savefig(plt,folder*"translations_pic5_bkg_sub.png")   
   #    save_translations(shifts2, file_sample[1:end-3]*"_bkg_sub")

   #    # RH approach
   #    data = load_data(file_sample)
   #    data_illu = similar(data)
   #    data_illu .= load_data(file_illu);
   #    shifts_rh, aligned_rh = get_translations_main_rh(data, data_illu, settings; red_dim=3, reg=1.0, pup_dia=nothing)
   #    @vr abs.(sum(aligned_rh, dims=3))

   # end

   # ##--- find axial shifts (only needs to be done once) ----------------------------------------------
   # # during the lateral movements of the sampel axial shifts are introduced to each measurement
   # # The following block tries to find said shifts (not optimized), which then can be used as a pre-propagation to the illumination/calibration field
   # if (false)
   #    data = load_data(file_sample, roi, range= 1:100)
   #    data_illu = similar(data)
   #    data_illu .= load_data(file_illu, roi, range=1:1);
   #    shifts2 = load_data(file_trans_mat, range=1:100) # read in hologramm of illumination holo
   #    # shift data to maximise possible roi
   #    data .= SpeckleReflections.shift.(data,  shifts2);
   #    data_illu .= SpeckleReflections.shift.(Ref(data_illu[1]), shifts2);

   #    roi_z = [2040:2240,700:900] # Define a region where there is always a reflective surface
   #    @vt data[100][roi_z[1], roi_z[2]] data_illu[100][roi_z[1], roi_z[2]]

   #    holo_params2 = get_holo_params(raw_sz, pad_factor; pup_dia=pup_dia, preserve_size=true)
   #    prop_init = 0.0
   #    dz = -0.025
   #    NZ = 30
   #    z_shifts_neg, corr_val_neg = find_axial_shift(data_illu, data, settings,holo_params2, NZ, dz; roi=roi_z, prop_init=prop_init)

   #    v = range(0.0, stop = (NZ-1) * dz, length = (NZ-1)) |> collect 
   #    z_vals = repeat([v], length(data))
   #    labels = range(1, length(data_illu), length=length(data_illu))
   #    plt = plot(z_vals[1], corr_val_pos[1], label=string(labels[1]))
   #    for idx in 2:100
   #       plot!(z_vals[idx], corr_val_pos[idx]) # label=string(labels[idx])
   #    end
   #    title!("Correlation of propagated illu.-field to obj.-field")
   #    xlabel!("prop. dist.[um]")
   #    ylabel!("Corr. value [-]")

   #    savefig(plt,folder*"corr_vals_pos_z.png")

   #    z_shifts_plt = []
   #    for idx in range(1,length(z_shifts_pos))
   #       append!(z_shifts_plt,z_shifts_pos[idx][1])
   #    end
   #    plt = plot(z_shifts_plt, title="Z-offset of 1. sampel pos to each other sampel pos.", label="prop. distance", xlabel="Pos. Nr.[-]", ylabel="prop. distance [um]")
      
   #    savefig(plt,folder*"Pos_axial_offsets_in_z_sample_to_mirror_M100.png")
 
   #    # save shifts in file
   #    h5open(folder*"z_shifts_mirror.h5", "w") do file
   #       for idx in eachindex(z_shifts_neg)
   #          write(file, string(idx), [z_shifts_neg[idx]...])
   #       end
   #    end
   # end

   ##--- read in data and shifts and preprocess data ----------------------------------------------
   # data = load_data(file_sample, [roi...,:])   
   # data_illu = similar(data)
   # data_illu .= load_data(file_sample, roi, range=1:1)
   # shifts2 = load_data(folder*"trans_values_extracted.h5", range=1:100) # read in hologramm of illumination holo
   # z_shifts = load_data(folder*"z_shifts_mirror.h5")
   # ref = gen_ref(data[1], settings)  
   # to load the data we need to apply shifts in real space and extract the hologram
   # This can be achieved by a preprocessing function applying a phase ramp in Fourier space
   # and extracting the hologram
   # if (false)  # If you use this less memory should be consumed.
   #    function preprocess_fct(dat, myshift)
   #       return rec_holo_shift(dat, holo_params, ref, myshift)
   #    end
   #    doscale(vec, scale) = vec .* [scale..., 1]
   #    scaled_shifts = doscale.(shifts2, Ref(raw_sz ./ size(data[1])))
   #    data_pr = preprocess_fct.(data, scaled_shifts);
   #    data_illu_pr = preprocess_fct.(data_illu, scaled_shifts);

   #    nsz = (size(data_pr[1]).*(holo_params.pad_fac, holo_params.pad_fac)..., length(data_pr))
   #    ave_img = zeros(nsz[1:2])
   #    ave_illu = zeros(nsz[1:2])
   #    for img_num in range(1, length(data_pr))
   #       ave_img .+= abs.(SpeckleReflections.resample(collect(SpeckleReflections.ift2d(data_pr[img_num])), nsz[1:2]));
   #       ave_illu .+= abs.(SpeckleReflections.resample(collect(SpeckleReflections.ift2d(data_illu_pr[img_num])), nsz[1:2]));
   #    end
   #    @vt ave_img./ave_illu

   #    # do pre propagation of illumination field if mirror/sampel was imaged out of focus
   #    data_illu_prop = similar(data_illu_pr)
   #    # mean_shift = sum(z_shifts)/length(z_shifts)
   #    new_samp_xy = ((settings.samp[1:2] .* size(data[1]) ./ size(data_pr[1])[1:2]))
   #    for img_num in range(1, length(data_illu_pr))
   #       Nz = 2
   #       prop_sampling = (new_samp_xy..., 2.5)
   #       PType = SpeckleReflections.complex_arr_type(typeof(SpeckleReflections.expand_dims(data_illu_pr[1], Val(3))))
   #       propagator = SpeckleReflections.propagator_col(PType, (size(data_illu_pr[1])[1:2]..., Nz), Float32.(prop_sampling), settings.lam; ref_idx = 1)
   #       aperture = SpeckleReflections.get_propagator_mask_3d(data_illu_pr[img_num], settings.lam, prop_sampling[1:2], settings.NA_det)
   #       data_illu_prop[img_num] =  data_illu_pr[img_num] .* conj.(propagator[:,:,2] .* aperture)
   #    end
   # else
   #    # shift, cut and then preprocess data
   #    function preprocess(dat)
   #       return rec_holo_shift(dat, holo_params, ref)
   #    end
   #    new_sz = size(data[1]).-470
   #    new_cent = new_sz .÷2
   #    data = SpeckleReflections.select_region.(SpeckleReflections.shift.(data,  shifts2); new_size=new_sz, center=new_cent);
   #    data_illu = SpeckleReflections.select_region.(SpeckleReflections.shift.(data_illu, shifts2); new_size=new_sz, center=new_cent);
   #    holo_params = get_holo_params(size(data[1]), pad_factor; pup_dia=pup_dia)
   #    ref = gen_ref(data[1], settings) 
   #    data_cr = preprocess.(data)
   #    data_illu_cr = preprocess.(data_illu)
   # end

   ##--- reconstruct on CPU ----------------------------------------------
   # @time rec_obj, phase_offsets_found = main_reconstruction(data, data_illu, settings, Nz; 
   #                                                                 phase_iterations=1);

end
