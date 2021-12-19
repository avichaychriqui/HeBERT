def spider_plot(df):
  '''
  retrieved from https://www.python-graph-gallery.com/391-radar-chart-with-several-individuals
  '''
  import matplotlib.pyplot as plt
  import pandas as pd
  from math import pi
      
  # number of variable
  categories=list(df)
  N = len(categories)
   
  # We are going to plot the first line of the data frame.
  # But we need to repeat the first value to close the circular graph:
  values=df.loc[0].values.flatten().tolist()
  values += values[:1]
  values
   
  # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
  angles = [n / float(N) * 2 * pi for n in range(N)]
  angles += angles[:1]
   
  # Initialise the spider plot
  fig, ax = plt.subplot(111, polar=True)
   
  # Draw one axe per variable + add labels
  plt.xticks(angles[:-1], categories, color='grey', size=8)
   
  # Draw ylabels
  ax.set_rlabel_position(0)
  plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
  plt.ylim(0,1)
   
  # Plot data
  ax.plot(angles, values, linewidth=1, linestyle='solid')
   
  # Fill area
  ax.fill(angles, values, 'b', alpha=0.1)
  
  # Show the graph
  return(fig, ax)
