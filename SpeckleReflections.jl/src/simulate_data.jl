
function do_simulation(obj, settings;
         Nimag = 100, # Number of images for averaging
         gc_period = 5,
         random_phases = false,
         reclaim_fct = GC.gc,
         num_photons = 0
      )
   ##-----3D Object-----##

   # generate Reference Mirror field
   ref = gen_ref(obj, settings); 

   phase_offset = zeros(Nimag)

   data_all = [];
   illu_all = [];
   # ref_all = [];

   # Main Simulation loop iterating through speckles
   for img_num in range(1,Nimag)
      println("processing image Nr.: $(img_num)/$(Nimag)")

      #------Generation------------
      # generate illumination (already Propagated into sample)
      illumination = gen_illumination(obj, settings); # = propagated illumination into each Object layer (3D Matrix)
      # 16 sec, 3 sec CUDA
      
      # illuminate object
      obj_illu = obj .* illumination;

      obj_prop = propagate_field(obj_illu, settings, settings.NA_det);   # each layer is propagated different distance according to depth
      obj_2D = sum(obj_prop, dims=3); # summ fields from different layers together at object surface
      illu_2D = @view illumination[:,:,1:1];

      # propagate field to detector via Lens system
      obj_L2 = obj_2D /  sqrt(500f0) #apply0_lens(obj_2D, r_pup_det)
      # illumination only
      illu_L2 = illu_2D / sqrt(10f0) # apply_lens(illu_2D, r_pup_det)

      #------Detection------------
      # interference with reference wave
      phase_offset[img_num] = (random_phases) ? rand(-pi:pi) : 0.0;

      holo_obj = normalize_intensity(abs2.((obj_L2 .* exp(1im*phase_offset[img_num])) .+ ref));
      holo_illu = normalize_intensity(abs2.(illu_L2 .+ ref)); # take first layer of speckle
      push!(data_all, Array(holo_obj[:,:,1]))
      push!(illu_all, Array(holo_illu[:,:,1]))
      # push!(ref_all, Array(ref))

      if (mod(img_num, gc_period) == 0)
         reclaim_fct();
      end
   end

   if (num_photons > 0)
      data_all = apply_noise.(data_all, num_photons)
      illu_all = apply_noise.(illu_all, num_photons)
   end

   return data_all, illu_all, phase_offset # ref_all, 
end

function apply_noise(data, n_photons=1000)
   return eltype(data).(Noise.poisson(Float64.(data .* n_photons./ maximum(data))))
end
