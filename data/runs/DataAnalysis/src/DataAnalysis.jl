module DataAnalysis

import CSV
import DataFrames
using Gnuplot

function get_csv_dataframe_from_directory(dir::String)::DataFrames.DataFrame
  return CSV.read("$(dir)/all.csv", DataFrames.DataFrame)
end

function plot_csv_dataframe(output_filename::String, df::DataFrames.DataFrame)
  num_runs::Int = length(unique(df.run))
  both::Bool = occursin("both", output_filename)
  visual_only::Bool = occursin("visual-only", output_filename)
  thermal_only::Bool = occursin("thermal-only", output_filename)
  title = ""
  if both
    title = "Visual and Thermal Heading Angle Versus Time"
  elseif visual_only
    title = "Visual Heading Angle Versus Time"
  elseif thermal_only
    title = "Thermal Heading Angle Versus Time"
  end
  colors = [
    "#5B9BD5", # Soft Blue
    "#8CCF88", # Soft Green
    "#F2C94C", # Warm Yellow
    "#A8A6D7", # Lavender
    "#FF6F61", # Coral
    "#B3B3B3", # Warm Grey
  ]
  @gp ylabel="Heading Angle" xlabel="Time [s]" title=title
  @gp :- "set key outside right center spacing 1.1"
  for i in 1:num_runs
    thisdf = df[df.run .== i, :]
    thisdf = DataFrames.sort(thisdf, :time_ms)
    # adjust time to be based off of first time
    thisdf.time_ms = thisdf.time_ms .- thisdf.time_ms[1]
    if both
      thisdf = thisdf[thisdf.heading_angle_v .!= "None", :]
      thisdf.heading_angle_v = parse.(Float32, thisdf.heading_angle_v)
      thisdf = thisdf[thisdf.heading_angle_t .!= "None", :]
      @gp :- thisdf.time_ms thisdf.heading_angle_v "w l tit 'run-$(i) visual theta' lw 2 lc '$(colors[i])'"
      @gp :- thisdf.time_ms thisdf.heading_angle_t "w l tit 'run-$(i) thermal theta' lw 2 dashtype 2 lc '$(colors[i])'"
    elseif visual_only
      thisdf = thisdf[thisdf.heading_angle_v .!= "None", :]
      thisdf.heading_angle_v = parse.(Float32, thisdf.heading_angle_v)
      @gp :- thisdf.time_ms thisdf.heading_angle_v "w l tit 'run-$(i) visual theta' lw 2 lc '$(colors[i])'"
    elseif thermal_only
      thisdf = thisdf[thisdf.heading_angle_t .!= "None", :]
      @gp :- thisdf.time_ms thisdf.heading_angle_t "w l tit 'run-$(i) thermal theta' lw 2 dashtype 1 lc '$(colors[i])'"
    end
  end
  filename = "$(split(output_filename, "/")[2]).png"
  Gnuplot.save(filename, term="pngcairo size 1000,600 fontscale 1.1")
end

function get_aggregated_data()
  dirs = ["../visual-only", "../thermal-only", "../both"]

  for d in dirs
    csv_dataframe = get_csv_dataframe_from_directory(d)
    plot_csv_dataframe(d, csv_dataframe)
  end
end

function main()
  get_aggregated_data()
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end

end # module DataAnalysis
