if (!window.dash_clientside) {
    window.dash_clientside = {}
}


    
window.dash_clientside.clientside = {
    update_store_data: function(rows, dataset, core, implementation, solver, log_or_lin, store) {
	/**
	 * Update timeseries figure when selected countries change,
	 * or type of cases (active cases or fatalities)
	 *
	 * Parameters
	 * ----------
	 *
	 *  rows: list of dicts
	 *	data of the table
	 *  selectedrows: list of indices
	 *	indices of selected countries
	 *  cases_type: str
	 *	active or death
	 *  log_or_lin: str
	 *	log or linear axis
	 *  store: list
	 *	store[0]: plotly-figure-dict, containing all the traces (all
	 *	countries, data and prediction, for active cases and deaths)
	 *	store[1]: list of countries to be used at initialization
	 */
	var fig = store;
	if (!rows) {
           throw "Figure data not loaded, aborting update."
       }
	var new_fig = {};
	new_fig['data'] = [];
	new_fig['layout'] = fig['layout'];
	var max = 0;
	var max_data = 0;
    var min = 1000000;
	var min_data = 0;
    //new_fig['layout']['annotations'][0]['visible'] = false;
    //new_fig['layout']['annotations'][1]['visible'] = true;
    for (i = 0; i < fig['data'].length; i++) {
    var name = fig['data'][i]['meta'][0];
        solver = "_" + solver + "_"
        if(dataset.includes("all")){
            dataset = ''
        }
        if(core.includes("all")){
            core = ''
        }
        if(implementation.includes("all")){
            implementation = ''
        }
        if(solver.includes("all")){
            solver = ''
        }
        if (name.includes(dataset) && name.includes(core) && name.includes(implementation) && name.includes(solver)){
            new_fig['data'].push(fig['data'][i]);
            max_data = Math.max(...fig['data'][i]['y']);
            min_data = Math.min(...fig['data'][i]['y']);
            if (max_data > max){
                max = max_data;
            }
            if (min_data < min){
                min = min_data;
            }
        }
    }
	
	new_fig['layout']['yaxis']['type'] = log_or_lin;
	if (log_or_lin === 'log'){
	    new_fig['layout']['legend']['x'] = .65;
	    new_fig['layout']['legend']['y'] = .1;
        if(max < 1){
            max = 1
        }
	    new_fig['layout']['yaxis']['range'] = [Math.log10(min) , Math.log10(max)];
	    new_fig['layout']['yaxis']['autorange'] = true;
	}
	else{
	    new_fig['layout']['legend']['x'] = .05;
	    new_fig['layout']['legend']['y'] = .8;
	    new_fig['layout']['yaxis']['autorange'] = true;
	}
        return new_fig;
    }
};