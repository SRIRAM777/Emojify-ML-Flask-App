$(document).ready(function() {

	$('#update-btn').click(function(event) {

		$('#update-class').text("Model is being updated. Kindly wait for a few seconds.")
		$.ajax({
			data : {
				dropdown_value : $('.dropdown-val').val(),
				sentence: $('#name-field').val()
			},
			type : 'POST',
			url : '/update'
		})
		.done(function(response) {
			$('#update-class').text(response.update_text)
			setTimeout( function() {console.log('Model updated')},70000)
			document.location.reload(true)
		});

		event.preventDefault();

	});

});