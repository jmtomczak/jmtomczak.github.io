
var grid_animating;

$(window).keypress(function (e) {
	if (e.which == 103)
	{
	
	if(!grid_animating)
	{
		grid_animating = true;
	
		if($("body.grid").length > 0)
		{
			$("body").removeClass("grid");
		}
		else {
			$("body").addClass("grid");
		}
	}
	}
});

function checkWindowSize() {
	
	if ( $(window).width() < 950 ) {
		$('body').addClass('small');
	}
	else {
		$('body').removeClass('small');
	}
	
}

$(window).resize(checkWindowSize);
$(window).load(checkWindowSize);