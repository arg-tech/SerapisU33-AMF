<?php
$baseurl = "http://amf2.arg.tech/";
$randomid = uniqid();

$tmpfile = $_FILES['fileToUpload']['tmp_name'];
$filename = basename($_FILES['fileToUpload']['name']);
$data = array(
    'file' => curl_file_create($tmpfile, $_FILES['fileToUpload']['type'], $filename)
);

$steps = array();

$turninator = $_POST['turninator_id'];
if ($turninator=="turninator-01") {
	array_push($steps, "turninator-01");
}

$propositionalizer = $_POST['propositionalizer_id'];
if ($propositionalizer=="propositionalizer-01") {
	array_push($steps, "propositionalizer-01");
}

$egmenter = $_POST['Segmenter_id'];
if ($egmenter=="Segmenter-01") {
	array_push($steps, "segmenter-01");
}

$selected_inference = $_POST['inference_id'];
if ($selected_inference=="DAM-01") {
	array_push($steps, "dam-01");
}
if ($selected_inference=="DAM-02") {
	array_push($steps, "dam-02");
}
if ($selected_inference=="TE") {
	array_push($steps, "bert-te");
}
if ($selected_inference=="DAM-03") {
	array_push($steps, "dam-03");
}

foreach ($steps as $step) {
	$purl = $baseurl . $step;
	$outfile = "tmp/" . $randomid . "-" . $step . ".json";

	$ch = curl_init();
	curl_setopt($ch, CURLOPT_URL, $purl);
	curl_setopt($ch, CURLOPT_TIMEOUT,500000000000); // 500 seconds
	curl_setopt($ch, CURLOPT_POST, 1);
	curl_setopt($ch, CURLOPT_HEADER, 0);
	curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1); 
	curl_setopt($ch, CURLOPT_USERAGENT, "Mozilla/4.0 (compatible;)");   
	curl_setopt($ch, CURLOPT_HTTPHEADER,array('Content-Type: multipart/form-data'));
	curl_setopt($ch, CURLOPT_FRESH_CONNECT, 1);   
	curl_setopt($ch, CURLOPT_FORBID_REUSE, 1);  
	curl_setopt($ch, CURLOPT_TIMEOUT, 500000000000);
	curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
	$result = curl_exec ($ch);

	if ($result === FALSE) {
	    echo "Error: " . curl_error($ch);
	    curl_close ($ch);
	}else{
	    $myfile = fopen($outfile, "w") or die("Unable to open file!");
	    fwrite($myfile, $result);
	    fclose($myfile); 
            $cfile = curl_file_create(realpath($outfile), 'application/json', 'stepjson');
            $data = array (
                'file' => $cfile
            );
	    curl_close ($ch);
	}
}

$target_url = "http://ws.arg.tech/t/json-svg";           
$cfile = curl_file_create(realpath($outfile), 'application/json', 'stepjson');

$post = array (
	'file' => $cfile
);    

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $target_url);
curl_setopt($ch, CURLOPT_POST, 1);
curl_setopt($ch, CURLOPT_HEADER, 0);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1); 
curl_setopt($ch, CURLOPT_USERAGENT, "Mozilla/4.0 (compatible;)");   
curl_setopt($ch, CURLOPT_HTTPHEADER,array('Content-Type: multipart/form-data'));
curl_setopt($ch, CURLOPT_FRESH_CONNECT, 1);   
curl_setopt($ch, CURLOPT_FORBID_REUSE, 1);  
curl_setopt($ch, CURLOPT_TIMEOUT, 100);
curl_setopt($ch, CURLOPT_POSTFIELDS, $post);

$result = curl_exec ($ch);

curl_close ($ch);

if ($result === FALSE) {
	$result = '<div id="pageloader" class="text-center">	  
				<img src="http://cdnjs.cloudflare.com/ajax/libs/semantic-ui/0.16.1/images/loader-large.gif" alt="Processing..." />
				<br>
				Processing...
			   </div>';
}
?>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <title>Argument Mining Framework</title>
    <link href="style.css" rel="stylesheet" type="text/css" />

    <link rel="stylesheet" type="text/css" href="/css/pretty-json.css" />

    <!-- lib -->
    <script type="text/javascript" src="/libs/jquery-1.11.1.min.js" ></script>
    <script type="text/javascript" src="/libs/underscore-min.js" ></script>
    <script type="text/javascript" src="/libs/backbone-min.js" ></script>

    <script type="text/javascript" src="/build/pretty-json-min.js" ></script>

</head>

<body>


 <header id="header" header-mobile-toggle="" class="primary-background">

    <div class="grid mx-l">
        <div>
            <a href="http://amf.arg.tech/" class="logo">
            	<img class="icon-image" src="http://amf.arg.tech/images/icon.png" alt="Logo">
                <span class="logo-text">Argument Mining Framework</span>
            </a>
	        <a href="http://arg.tech"><div class="mobile-menu-toggle hide-over-l"><svg class="svg-icon" data-icon="more" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    			<path d="M0 0h24v24H0z" fill="none"></path>
			    <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"></path>
			</svg></div></a>
        </div>
			
        <div class="header-search hide-under-l"></div>

        <div class="text-right">
            <div class="header-links">
                <div class="links text-center">
                	<a href="http://arg.tech">
                		<img class="logo-image" src="http://amf.arg.tech/images/arg-tech-w-s.png">
					</a>
                </div>
			   
             </div>
        </div>
    </div>
 </header>

 <div id="main">

    <div class="container px-xl py-s">
        <div class="block">
			The system here connects together a series of argument mining web services each of which is available at <a href="http://ws.arg.tech">http://ws.arg.tech</a> as a part of the web service API provided by the Centre for Argument Technology.  
			<br> &nbsp;
        </div>
        <div class="icon-list block">
                  </div>
	</div>
 
	<div class="container" id="home-default">
        <div class="grid half gap-xxl no-row-gap">
            <div>
                <div id="text" class="card mb-xl">
                    <h3 class="card-title">JSON Result</h3>
					<div class="text-center">
						<label class="button outline" id="expandAll">Expand All</label>
						<label class="button outline" id="showCode">Show Code</label>
					</div>
                    <div class="px-m">
	                    <textarea id="input" class="hidden"></textarea>
						<span id="result"></span>
                    </div>
                </div>
            </div>


            <div>
	            <div class="card mb-xl">
		            <h3 class="card-title">Argument Map</h3>
                    <div class="px-m py-xxl">
                    	<?php echo $result;?>
			        </div>
	            </div>
            </div>
        </div>
	  <div class="block">	
	  	<div class="text-center">	  
			 <a href="http://amf.arg.tech/" class="logo"><span class="button">Return</span></a>
		</div>
	  </div>
	</div>
</div> 	

<div id="footer">
	<div class="container">
    </div>
</div>

<script>
    $(document).ready(function() {
	
	var el = {
		input: $('#input'),
        result: $('#result')
    };

    var result = '<?php echo $result;?>'; 

    el.input.val(JSON.stringify(result,null,4));
	
	var json = el.input.val();
    var data;
    try{ data = JSON.parse(json); }
		catch(e){ 
        	alert('not valid JSON');
            return;
    	}
        var node = new PrettyJSON.view.Node({ 
	        el:el.result,
            data: data,
            dateFormat:"DD/MM/YYYY - HH24:MI:SS"
        });
        
        var expanded = false;
            
        document.getElementById("expandAll").onclick = function () {
			if (expanded) {
				node.collapseAll();
		   		this.innerHTML = " Expand All ";
		   		expanded = false;
			} else {
				node.expandAll();
		   		this.innerHTML = "Collapse All";
		   		expanded = true;
			}
		};

		var visible = false;
        document.getElementById("showCode").onclick = function () {
			if (visible) {
				el.input.hide();
		   		this.innerHTML = "Show Code";
		   		visible = false;
			} else {
				el.input.show();
		   		this.innerHTML = "Hide Code";
		   		visible = true;
			}
		};

    });
</script>

	    	 

</body>
</html> 





    




