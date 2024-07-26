
# Spatiotemporal Analysis of FLUXNET Data: SWC and RECO
My submission for the Spatiotemporal Data class, summer 2024. 
## Introduction

This project analyses Soil Water Content (SWC) and Ecosystem Respiration (RECO) data from the [FLUXNET](https://fluxnet.org/) network. FLUXNET is a global network of micrometeorological tower sites that measure the exchanges of carbon dioxide, water vapour, and energy between terrestrial ecosystems and the atmosphere.

SWC is a key component of the hydrological cycle, influencing water availability for plants and soil processes. RECO represents the total respiration from an ecosystem, including plants, animals, and microorganisms. By examining these variables together, we aim to gain insights into how water availability influences carbon fluxes and overall ecosystem functioning.

## Goal and Results

The primary goals of this analysis are:

1. To explore the relationship between SWC and RECO across various FLUXNET stations and soil depths.
2. To investigate seasonal variations in SWC and RECO.
3. To examine the correlation between SWC and RECO, considering the non-linear nature of their relationship.


Key results include:
- predominately weak-negative correlation between SWC and RECO
- Low R squared between SWC and RECO
- Location plays a role in relationship

## Methods

This analysis involves the following steps:

1. Data preprocessing: Cleaning and organising FLUXNET data for SWC and RECO.
2. Time series analysis: Examining temporal patterns in both variables.
3. Correlation analysis: Investigating the relationship between SWC and RECO at different soil depths.
4. Seasonal variation: Analysing seasonal patterns in the data.
5. Visualisation: Creating interactive plots to display results.

I use Python for data processing and analysis, with libraries such as Pandas for data manipulation and Plotly for interactive visualisations.


## Code Structure


- `Dockerfile`: The Dockerfile compiles a container with all necessary dependencies to run the analysis. It ensures reproducibility across different environments.
- `report.qmd`: Quarto document containing the full analysis report and results.
- `docker-compose.yml`: Configures the Docker services for the project.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `ci.yml`: Continuous integration configuration file.
- `requirements.txt`: Lists python dependencies required for the project.
- `README.md`: This file provides an overall description of the entire project.
- `data_prcoessing_functions.py`: contains most (not all) of the functions I use in the report. The functions are also found in the report, but I load some first from this file, since I was having some strange problems with compiling when all the functions were in the code.
- `data /RECO/`: the data used in this analysis, from FLUXNET
- `/data/SWC/` : the data used in the analysis, from FLUXNET
- `/data/fluxnet-station-coords.csv`: A CSV file containing all available station location data from the FLUXNET towers
- `/data/filtered-fluxnet-station-coords.csv`: The same as above, filtered by me (code shown in report) for only the stations I need for the analysis.


## Acknowledgments

This project uses data from the FLUXNET network. I acknowledge the FLUXNET community for their continued efforts in providing this valuable dataset.