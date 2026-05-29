async function predictLoan() {

    const data = {
        Age: document.getElementById("Age").value,
        Income: document.getElementById("Income").value,
        LoanAmount: document.getElementById("LoanAmount").value,
        CreditScore: document.getElementById("CreditScore").value,
        MonthsEmployed: document.getElementById("MonthsEmployed").value,
        NumCreditLines: document.getElementById("NumCreditLines").value,
        InterestRate: document.getElementById("InterestRate").value,
        LoanTerm: document.getElementById("LoanTerm").value,
        DTIRatio: document.getElementById("DTIRatio").value,
        Education: document.getElementById("Education").value,
        EmploymentType: document.getElementById("EmploymentType").value,
        MaritalStatus: document.getElementById("MaritalStatus").value,
        HasMortgage: document.getElementById("HasMortgage").value,
        HasDependents: document.getElementById("HasDependents").value,
        LoanPurpose: document.getElementById("LoanPurpose").value,
        HasCoSigner: document.getElementById("HasCoSigner").value
    };

    const response = await fetch(
        "/loan/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        }
    );

    const result = await response.json();

    document.getElementById("result").innerHTML =
        result.prediction;
}