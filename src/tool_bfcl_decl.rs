#[derive(Clone)]
pub struct BfclDatasetEntry{
    pub id: String,
    pub question: String,
    pub functions: Vec<BfclFunctionDef>,
}

impl BfclDatasetEntry {
    pub fn try_from(raw_entry: serde_json::Value) -> Result<Self, String> {
        
    }
}


#[derive(Clone)]
pub struct BfclFunctionDef{
    pub name: String,
    pub description: String,
    pub parameters: Vec<BfclParameter>,
}

#[derive(Clone)]
pub struct BfclParameter{
    pub name: String,
    pub r#type: String,
    pub description: String,
}