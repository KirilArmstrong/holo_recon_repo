
function do_simulation(obj, settings;
         Nlam = length(settings.lam), # Number of images for averaging
         gc_period = 5,
         random_phases = false,
         reclaim_fct = GC.gc,
         num_photons = 0
      )
   ##-----3D Object-----##
   # generate Reference Mirror field
   # only one phase offset since we have only one position of the sample here
   phase_offset = 0 #zeros(Nlam)
   phase_offset = (random_phases) ? rand(-pi:pi) : 0.0; # this line needs to be changed??

   data_all = zeros((size(obj)[1:2]...,Nlam));
   illu_all = similar(data_all);

   fna = 3 .*[-settings.NA_det, settings.NA_det]; # tilt diagonal -> into settings?
   k_ref = fna[1].*settings.samp[1].*settings.n./settings.lam # vector of k_refs

   # create new random speckle pattern
   phi_spek = similar(obj, size(obj)[1:2]); # RH
   rand!(phi_spek); # RH
   phi_spek .*= 2pi;
   phi_spek .-= pi; # make random NxN matrix with range [-pi, pi]
   
   # NEW: Main Simulation loop iterating through wavelength spectrum
   for lam_num in range(1,Nlam)
      println("processing wavelength Nr.: $(lam_num)/$(Nlam)")
      freq_ref = k_ref[lam_num]/1.5
      ref = gen_ref(obj, [freq_ref, freq_ref])

      #------Generation------------
      # generate illumination (already Propagated into sample)
      illumination = gen_illumination(obj, settings.lam[lam_num], settings; phi_spek=phi_spek); # = propagated illumination into each Object layer (3D Matrix)
      
      # illuminate object
      obj_illu = obj .* illumination;

      obj_prop = propagate_field(obj_illu, settings.lam[lam_num], settings, settings.NA_det);   # each layer is propagated different distance according to depth
      obj_2D = sum(obj_prop, dims=3); # summ fields from different layers together at object surface
      illu_2D = @view illumination[:,:,1:1];

      # propagate field to detector via Lens system
      obj_L2 = obj_2D /  sqrt(500f0) #apply0_lens(obj_2D, r_pup_det)
      # illumination only
      illu_L2 = illu_2D / sqrt(10f0) # apply_lens(illu_2D, r_pup_det)

      #------Detection------------
      # interference with reference wave

      holo_obj = normalize_intensity(abs2.((obj_L2 .* exp(1im*phase_offset)) .+ ref));
      holo_illu = normalize_intensity(abs2.(illu_L2 .+ ref)); # take first layer of speckle
      data_all[:,:,lam_num] .= Array(holo_obj[:,:,1])
      illu_all[:,:,lam_num] .=  Array(holo_illu[:,:,1])
      # push!(ref_all, Array(ref))

      if (mod(lam_num, gc_period) == 0)
         reclaim_fct();
      end

   end

   if (num_photons > 0)
      data_all = apply_noise.(data_all, num_photons)
      illu_all = apply_noise.(illu_all, num_photons)
   end

   return [data_all], [illu_all], [phase_offset] # ref_all, 
end

function apply_noise(data, n_photons=1000)
   return eltype(data).(Noise.poisson(Float64.(data .* n_photons./ maximum(data))))
end
