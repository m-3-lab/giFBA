import cobra as cb
import numpy as np
import pandas as pd
from cobra.util.solver import linear_reaction_coefficients
from . import utils
from .config import GROWTH_MIN_OBJ, ROUND

class CommunitySummary:
    """Class to summarize the results of giFBA analysis.
    
    Attributes:
        
    """
    def __init__(self, community, iteration_shown=None, element="C"):
        
        # initialize attributes
        self.iteration_shown = None
        self.method = None
        self.objective_rxns = None
        self.objective_vals = None
        self.objective_total = None
        self.uptake = None
        self.secretion = None
        self.element = element
        
        self._build_summary_frames(community, iteration_shown)

    def _build_summary_frames(self, community, iteration_shown):
        # check iteration_shown is valid
        if self.iteration_shown is not None:
            if not isinstance(self.iteration_shown, (int, float)) or self.iteration_shown < 0 or self.iteration_shown >= self.n_iterations:
                raise ValueError("iteration_shown must be a non-negative integer less than the number of iterations.")
            else:
                self.iteration_shown = int(iteration_shown)
        else: 
            self.iteration_shown = community.n_iterations - 1

        # pull organism fluxes
        self.community = community
        self.flux = self.community.org_final.copy()

        # extract objectives and create expressions to print
        self.method = self.community.method
        self.objective_rxns = self.community.objective_rxns
        self.objective_vals = [self.flux.loc[model_idx, rxn] for model_idx, rxn in self.objective_rxns.items()]
        self.objective_expressions = [f"1.0 * {rxn} = {self.objective_vals[model_idx]}" for model_idx, rxn in self.objective_rxns.items()]

        # calculate total objective value
        self.objective_total = np.array(self.objective_vals).sum()
        self.objective_total_expression = f"Sum(Model_i Biomass) = {self.objective_total}"

        # create summary dataframe for overall community
        self.total_flux = self.community.org_final[self.community.exchange_ids].sum()
        self.total_flux = self.total_flux.T.reset_index()
        self.total_flux = self.total_flux.copy()
        self.total_flux.columns = ["Exchange", "Flux"]

        # add metabolite to env_flux
        self.total_flux["Metabolite"] = self.total_flux["Exchange"].map(self.community.exchange_to_metabolite_id)
        self.total_flux = self.total_flux.set_index("Metabolite")


        # add element information
        metabolites = {m.id: m for m in self.community.exchange_metabolites}
        self.total_flux[f"{self.element}-Number"] = [
            metabolites[met_id].elements.get(self.element, 0) if met_id in metabolites else 0
            for met_id in self.total_flux.index
        ]
        self.total_flux[f"{self.element}-Flux"] = self.total_flux[f"{self.element}-Number"] * self.total_flux["Flux"].abs()

        # remove unused fluxes
        self.total_flux = self.total_flux[self.total_flux['Flux'] != 0] # remove zero fluxes

        # create dfs for organisms
        self.flux = self.flux[self.community.exchange_ids].copy()
        self.flux = self.flux.reset_index()
        self.flux.columns = ["Model"] + list(self.flux.columns[1:])
        self.flux = pd.melt(
            self.flux, 
            id_vars=["Model"], 
            var_name="Exchange", 
            value_name="Flux"
        )
        self.flux["Metabolite"] = self.flux["Exchange"].map(self.community.exchange_to_metabolite_id)
        self.flux["Metabolite"] = self.flux["Metabolite"].fillna(self.flux["Exchange"])
        self.flux = self.flux.set_index(["Model", "Exchange"])

        # add element information
        metabolites = {m.id if pd.notnull(m.id) else m: m for m in self.community.exchange_metabolites}
        self.flux[f"{self.element}-Number"] = [
            metabolites[met_id].elements.get(self.element, 0) if met_id in metabolites else 0
            for met_id in self.flux["Metabolite"]
        ]
        self.flux[f"{self.element}-Flux"] = self.flux[f"{self.element}-Number"] * self.flux["Flux"].abs()        

        # remove unused fluxes
        self.flux = self.flux[self.flux['Flux'] != 0] # remove zero fluxes

        return
    
    def to_cytoscape(self):
        # pull pertinent info for cytoscape edge table
        self.cytoscape_edges = self.flux.reset_index()
        self.cytoscape_edges["Source"] = self.cytoscape_edges["Model"].map(self.community.model_names)
        self.cytoscape_edges["Target"] = self.cytoscape_edges["Metabolite"]
        self.cytoscape_edges["Type"] = ["Uptake" if flux < 0 else "Secretion" for flux in self.cytoscape_edges["Flux"]]
        self.cytoscape_edges["Value"] = self.cytoscape_edges["Flux"].abs()

        # drop all other info
        self.cytoscape_edges = self.cytoscape_edges[["Source", "Target", "Type", "Value"]]

        # create cytoscape node table
        self.cytoscape_nodes = pd.DataFrame()
        self.cytoscape_nodes["ID"] = pd.concat([self.cytoscape_edges["Source"], self.cytoscape_edges["Target"]]).unique()
        self.cytoscape_nodes["Name"] = [self.community.metabolite_id_to_name.get(id, id) for id in self.cytoscape_nodes["ID"]]
        self.cytoscape_nodes["Type"] = ["Organism" if id in self.community.model_names.values() else "Metabolite" for id in self.cytoscape_nodes["ID"]]

        return self.cytoscape_edges, self.cytoscape_nodes

    def to_string(self):
        """Display the summary of the community."""
        output = []
        output.append(f"Community Summary (Cumulative through Iteration {self.iteration_shown}):\n")
        output.append(f"Optimization Type: {self.method}\n")
        output.append(f"{self.objective_total_expression}\n\n")

        # uptake
        output.append("Uptake:\n")
        uptake = self.total_flux.loc[self.total_flux['Flux'] < 0].copy()
        uptake_total_element_flux = uptake.loc[:, f"{self.element}-Flux"].sum()
        if uptake_total_element_flux > 0:
            uptake.loc[:, f"{self.element}-Flux"] = uptake.loc[:, f"{self.element}-Flux"] / uptake_total_element_flux * 100
        else:
            uptake.loc[:, f"{self.element}-Flux"] = 0
        uptake["Flux"] = uptake["Flux"].abs()
        uptake[f"{self.element}-Flux"] = uptake[f"{self.element}-Flux"].map("{:.2f}%".format)
        output.append(f"{uptake.reset_index().to_string(index=False)}\n\n")

        # secretion
        output.append("Secretion:\n")
        secretion = self.total_flux.loc[self.total_flux['Flux'] > 0].copy()
        secretion_total_element_flux = secretion.loc[:, f"{self.element}-Flux"].sum()
        if secretion_total_element_flux > 0:
            secretion.loc[:, f"{self.element}-Flux"] = secretion.loc[:, f"{self.element}-Flux"] / secretion_total_element_flux * 100
        else:
            secretion.loc[:, f"{self.element}-Flux"] = 0
        secretion[f"{self.element}-Flux"] = secretion[f"{self.element}-Flux"].map("{:.2f}%".format)
        output.append(f"{secretion.reset_index().to_string(index=False)}\n\n")

        for model_idx in self.flux.index.get_level_values(0).unique():
            output.append("-----------------------------------------------------------------\n")
            output.append(f"{self.community.model_names[model_idx]} (Model {model_idx}) Summary:\n")
            output.append(f"{self.objective_expressions[model_idx]}\n\n")

            # uptake
            output.append(f"{self.community.model_names[model_idx]} Uptake:\n")
            uptake = self.flux.loc[model_idx][self.flux.loc[model_idx]['Flux'] < 0].copy()
            uptake["Flux"] = uptake["Flux"].abs()
            uptake_total_element_flux = uptake.loc[:, f"{self.element}-Flux"].sum()
            if uptake_total_element_flux > 0:
                uptake.loc[:, f"{self.element}-Flux"] = uptake.loc[:, f"{self.element}-Flux"] / uptake_total_element_flux * 100
            else:
                uptake.loc[:, f"{self.element}-Flux"] = 0
            uptake[f"{self.element}-Flux"] = uptake[f"{self.element}-Flux"].map("{:.2f}%".format)
            output.append(f"{uptake.reset_index().to_string(index=False)}\n\n")

            # secretion
            output.append(f"Model {model_idx} Secretion:\n")
            secretion = self.flux.loc[model_idx][self.flux.loc[model_idx]['Flux'] > 0].copy()
            secretion_total_element_flux = secretion.loc[:, f"{self.element}-Flux"].sum()
            if secretion_total_element_flux > 0:
                secretion.loc[:, f"{self.element}-Flux"] = secretion.loc[:, f"{self.element}-Flux"] / secretion_total_element_flux * 100
            else:
                secretion.loc[:, f"{self.element}-Flux"] = 0
            secretion[f"{self.element}-Flux"] = secretion[f"{self.element}-Flux"].map("{:.2f}%".format)
            output.append(f"{secretion.reset_index().to_string(index=False)}\n\n")
        output.append("\nAccessible at summary.flux or summary.total_flux for cumulative fluxes.\n")

        return "".join(output)

    def __str__(self):
        """Return the string representation of the summary."""
        return self.to_string()
    
    def __repr__(self):
        """Return the string representation of the summary."""
        return self.to_string()

    def _repr_html_(self):
        html = f"<h3>Community Summary (Cumulative through Iteration {self.iteration_shown})</h3>"
        html += f"<b>Optimization Type:</b> {self.method}<br>"
        html += f"{self.objective_total_expression}<br>"

        # Community Uptake Table
        html += "<h4>Community Uptake</h4>"
        uptake = self.total_flux[self.total_flux['Flux'] < 0].copy()
        uptake['Flux'] = uptake['Flux'].abs()
        if not uptake.empty:
            uptake_total_element_flux = uptake.loc[:, f"{self.element}-Flux"].sum()
            if uptake_total_element_flux > 0:
                uptake.loc[:, f"{self.element}-Flux"] = uptake.loc[:, f"{self.element}-Flux"] / uptake_total_element_flux * 100
            else:
                uptake.loc[:, f"{self.element}-Flux"] = 0
            uptake[f"{self.element}-Flux"] = uptake[f"{self.element}-Flux"].map("{:.2f}%".format)
            html += uptake.reset_index().to_html(index=False)
        else:
            html += "<i>No uptake fluxes</i>"

        # Community Secretion Table
        html += "<h4>Community Secretion</h4>"
        secretion = self.total_flux[self.total_flux['Flux'] > 0].copy()
        if not secretion.empty:
            secretion_total_element_flux = secretion.loc[:, f"{self.element}-Flux"].sum()
            if secretion_total_element_flux > 0:
                secretion.loc[:, f"{self.element}-Flux"] = secretion.loc[:, f"{self.element}-Flux"] / secretion_total_element_flux * 100
            else:
                secretion.loc[:, f"{self.element}-Flux"] = 0
            secretion[f"{self.element}-Flux"] = secretion[f"{self.element}-Flux"].map("{:.2f}%".format)
            html += secretion.reset_index().to_html(index=False)
        else:
            html += "<i>No secretion fluxes</i>"

        # Organism-level tables
        for model_idx in self.flux.index.get_level_values(0).unique().sort_values():
            html += f"<hr><h4>{self.community.model_names[model_idx]} (Model {model_idx}) Summary</h4>"
            html += f"{self.objective_expressions[model_idx]}<br>"

            # Organism Uptake Table
            org_uptake = self.flux.loc[model_idx][self.flux.loc[model_idx]['Flux'] < 0].copy()
            org_uptake['Flux'] = org_uptake['Flux'].abs()
            uptake_total_element_flux = org_uptake.loc[:, f"{self.element}-Flux"].sum()
            if uptake_total_element_flux > 0:
                org_uptake.loc[:, f"{self.element}-Flux"] = org_uptake.loc[:, f"{self.element}-Flux"] / uptake_total_element_flux * 100
            else:
                org_uptake.loc[:, f"{self.element}-Flux"] = 0
            org_uptake[f"{self.element}-Flux"] = org_uptake[f"{self.element}-Flux"].map("{:.2f}%".format)
            html += f"<b>Model {model_idx} Uptake:</b>"
            if not org_uptake.empty:
                html += org_uptake.reset_index().to_html(index=False)
            else:
                html += "<i>No uptake fluxes</i>"

            # Organism Secretion Table
            org_secretion = self.flux.loc[model_idx][self.flux.loc[model_idx]['Flux'] > 0].copy()
            secretion_total_element_flux = org_secretion.loc[:, f"{self.element}-Flux"].sum()
            if secretion_total_element_flux > 0:
                org_secretion.loc[:, f"{self.element}-Flux"] = org_secretion.loc[:, f"{self.element}-Flux"] / secretion_total_element_flux * 100
            else:
                org_secretion.loc[:, f"{self.element}-Flux"] = 0
            org_secretion[f"{self.element}-Flux"] = org_secretion[f"{self.element}-Flux"].map("{:.2f}%".format)
            html += f"<b>Model {model_idx} Secretion:</b>"
            if not org_secretion.empty:
                html += org_secretion.reset_index().to_html(index=False)
            else:
                html += "<i>No secretion fluxes</i>"

        html += "<p>Accessible at summary.flux or summary.total_flux for cumulative fluxes.</p>"

        return html

    

