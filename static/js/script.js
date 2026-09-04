/**
 * AyurHerb - Interactive Client-side Script (script.js)
 */

document.addEventListener("DOMContentLoaded", function () {
  // Symptom chip toggling logic
  const symptomChips = document.querySelectorAll(".symptom-chip");
  const symptomsInput = document.getElementById("symptoms-input");
  const clearBtn = document.getElementById("clear-symptoms-btn");

  if (symptomChips.length > 0 && symptomsInput) {
    symptomChips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        const symptom = this.getAttribute("data-symptom") || this.innerText.trim();
        let currentText = symptomsInput.value.trim();
        
        let terms = currentText
          ? currentText.split(",").map(t => t.trim()).filter(t => t.length > 0)
          : [];

        const index = terms.findIndex(t => t.toLowerCase() === symptom.toLowerCase());

        if (index > -1) {
          // Deselect chip
          terms.splice(index, 1);
          this.classList.remove("active");
        } else {
          // Select chip
          terms.push(symptom);
          this.classList.add("active");
        }

        symptomsInput.value = terms.join(", ");
      });
    });

    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        symptomsInput.value = "";
        symptomChips.forEach(c => c.classList.remove("active"));
        symptomsInput.focus();
      });
    }

    // Sync chips when user types manually
    symptomsInput.addEventListener("input", function () {
      const currentText = this.value.toLowerCase();
      symptomChips.forEach(function (chip) {
        const symptom = (chip.getAttribute("data-symptom") || chip.innerText).toLowerCase();
        if (currentText.includes(symptom)) {
          chip.classList.add("active");
        } else {
          chip.classList.remove("active");
        }
      });
    });
  }

  // Auto-dismiss alerts after 5 seconds
  const alerts = document.querySelectorAll(".alert-dismissible");
  alerts.forEach(function (alert) {
    setTimeout(function () {
      const bsAlert = new bootstrap.Alert(alert);
      bsAlert.close();
    }, 5000);
  });
});
