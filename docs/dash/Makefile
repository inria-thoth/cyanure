
html:
	# Now build the app
	export DEBUG=False && python3 app.py &
	sleep 60
	wget -r http://127.0.0.1:8050/ 
	wget -r http://127.0.0.1:8050/_dash-layout 
	wget -r http://127.0.0.1:8050/_dash-dependencies
	sed -i 's+_dash-layout+cyanure/_dash-layout.json+g' 127.0.0.1:8050/_dash-component-suites/dash/dash-renderer/build/*.js 
	sed -i 's+_dash-dependencies+cyanure/_dash-dependencies.json+g' 127.0.0.1:8050/_dash-component-suites/dash/dash-renderer/build/*.js
	# Add our head
	sed -i '/<head>/ r head.html' 127.0.0.1:8050/index.html
	mv 127.0.0.1:8050/_dash-layout 127.0.0.1:8050/_dash-layout.json	
	mv 127.0.0.1:8050/_dash-dependencies 127.0.0.1:8050/_dash-dependencies.json
	cp thumbnail.png 127.0.0.1:8050/
	cp -r csvs 127.0.0.1:8050/
	cp assets/* 127.0.0.1:8050/assets/
	cp _static/async* 127.0.0.1:8050/_dash-component-suites/dash/dcc/
	cp _static/async-table* 127.0.0.1:8050/_dash-component-suites/dash/dash_table/
	sed -i 's+_dash-component-suites+cyanure/_dash-component-suites+g' 127.0.0.1:8050/index.html 
	sed -i 's+assets+cyanure/assets+g' 127.0.0.1:8050/index.html 
	mv 127.0.0.1:8050/index.html 127.0.0.1:8050/dash.html
	ps | grep python | awk '{print $$1}' | xargs kill -9	
