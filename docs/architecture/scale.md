ok, so yes i agree that we should allocate the array size in advance, let's give ourselves enough so that we can go from the level of a quark/gluon to   
  the entire observable universe. how many do we need                                                                                                      
                                                                                                                                                           
⏺ Quark/gluon scale: ~10⁻¹⁸ meters                                                                                                                         
  Observable universe: ~10²⁷ meters                                                                                                                        
                                                                                                                                                           
  Ratio: ~10⁴⁵                                                                                                                                             
                                          
  Each layer multiplies by 3. We need 3ᴺ ≥ 10⁴⁵:                                                                                                           
                                                                                                                                                           
  N = 45 / log₁₀(3) = 45 / 0.477 ≈ 94 levels                                                                                                               
                                                                                                                                                           
  Some landmarks along the way:                                                                                                                            
                                                                                                                                                           
  ┌───────┬──────────┬─────────────────────────────────┐          
  │ Layer │  Scale   │         What lives here         │                                                                                                   
  ├───────┼──────────┼─────────────────────────────────┤          
  │ 0     │ 10⁻¹⁸ m  │ Quarks                          │                                                                                                   
  ├───────┼──────────┼─────────────────────────────────┤
  │ 12    │ ~10⁻¹² m │ Atoms                           │                                                                                                   
  ├───────┼──────────┼─────────────────────────────────┤          
  │ 25    │ ~10⁻⁶ m  │ Cells, bacteria                 │
  ├───────┼──────────┼─────────────────────────────────┤                                                                                                   
  │ 38    │ ~1 m     │ Player blocks (Minecraft scale) │
  ├───────┼──────────┼─────────────────────────────────┤                                                                                                   
  │ 50    │ ~10⁶ m   │ Planets                         │                                                                                                   
  ├───────┼──────────┼─────────────────────────────────┤
  │ 58    │ ~10⁹ m   │ Solar systems                   │                                                                                                   
  ├───────┼──────────┼─────────────────────────────────┤          
  │ 70    │ ~10¹⁵ m  │ Interstellar                    │                                                                                                   
  ├───────┼──────────┼─────────────────────────────────┤          
  │ 82    │ ~10²¹ m  │ Galaxies                        │                                                                                                   
  ├───────┼──────────┼─────────────────────────────────┤
  │ 94    │ ~10²⁷ m  │ Observable universe             │                                                                                                   
  └───────┴──────────┴─────────────────────────────────┘          
                                                                                                                                                           
  So MAX_DEPTH = 96 gives quarks to the observable universe with a couple levels of headroom. The path array is 96 bytes — fits in a cache line and a half.
   The depth: u8 field handles it fine (max 255).                                                                                                          
                                                                  
  pub const MAX_DEPTH: usize = 96;                                                                                                                         
   
❯ what is a cache line and a half? is it going to add a lot of overhead to do 96? maybe we should start with something like 30 if the paths become a       
  limiting factor in computation time?                                                                                                                     
                                                                                                                                                           
⏺ A cache line is 64 bytes — the unit the CPU loads from memory. 96 bytes straddles two cache lines, so every path access potentially loads 128 bytes from 
  RAM. Not catastrophic, but not free.                                                                                                                     
                                                                                                                                                           
  The real overhead is that paths are everywhere in hot loops:                                                                                             
  - Every WalkFrame on the DFS stack carries a path
  - Every Position carries a path                                                                                                                          
  - Paths get copied, hashed, and compared constantly                                                                                                      
  - 96 bytes vs 30 bytes means ~3× more data shuffled through all of those
                                                                                                                                                           
  But honestly, the difference between 30 and 96 bytes per path is unlikely to be the bottleneck. The walk stack might have a few hundred frames. That's
  30KB vs 10KB — both fit in L1 cache.                                                                                                                     
                                                                                                                                                           
  Start with 30. 3³⁰ ≈ 2 × 10¹⁴ — that's 200 trillion resolution per axis. More than enough for planet-scale gameplay with room to spare. Bumping it later 
  is a one-line constant change. No code depends on the specific value.                                                                                    
                                                                                                                                                           
  pub const MAX_DEPTH: usize = 30;                                                                                                                         
                                                                  
  If paths ever become a bottleneck, the fix isn't shrinking the array — it's not carrying full paths in the hot loop at all (the walk stack could track   
  depth + parent index instead of copying the full path each frame). But that's an optimization for later.