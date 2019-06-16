$(document).ready(function() {

	$('form').on('submit', function(event) {

		$.ajax({
			data : {
				name : $('#name-field').val()
			},
			type : 'POST',
			url : '/predict'
		})
		.done(function(response) {

			if(response.prediction == 0)
				{
					$("#greeting").text("🙂" + " Happy Face");
				}
				else if(response.prediction == 1)
				{
					$("#greeting").text("😨" + " Fearful face");
				}
				else if(response.prediction == 2)
				{
					$("#greeting").text("😠" + " Anger");
				}
				else if(response.prediction == 3)
				{
					$("#greeting").text("😥" + " Sadness");
				}
				else if(response.prediction == 4)
				{
					$("#greeting").text("😖" + " Disgust");
				}
				else if(response.prediction == 5)
				{
					$("#greeting").text("😞" + " Shame");
				}
				else if(response.prediction == 6)
				{
					$("#greeting").text("😰" + " Guilt");
				}
				console.log(response);

		});

		event.preventDefault();

	});

});