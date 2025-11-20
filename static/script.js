function fetchDiseaseData() {
  $.getJSON("/get_disease_data", function (data) {
    $("#healthy-count").text(data.healthy_count);
    $("#diseased-count").text(data.diseased_count);

    const total = data.healthy_count + data.diseased_count;
    $("#totalScanned").text(total);

    const healthRate = total > 0 ? Math.round((data.healthy_count / total) * 100) : 0;
    $("#healthPercentage").text(healthRate + "%");
    $("#goodPercentage").text(healthRate + "% of total");
    $("#diseasedPercentage").text((100 - healthRate) + "% of total");

    $("#progressFill").css("width", healthRate + "%");
  });
}

// refresh every 2 seconds
setInterval(fetchDiseaseData, 2000);
