from backend.app.ml_service import get_ml_service

def check_mongodb_features():
    ml_service = get_ml_service()
    db = ml_service.db

    # Get one document
    doc = db.ml_features.find_one()
    if not doc:
        print("No documents found in ml_features collection")
        return

    print("=== MONGODB FEATURES CHECK ===")
    print(f"Total documents: {db.ml_features.count_documents({})}")

    # Check for new features
    has_publish_hour = 'publish_hour' in doc
    has_video_age_proxy = 'video_age_proxy' in doc

    print(f"Has publish_hour: {has_publish_hour}")
    print(f"Has video_age_proxy: {has_video_age_proxy}")

    if has_publish_hour:
        print(f"publish_hour value: {doc['publish_hour']}")
    if has_video_age_proxy:
        print(f"video_age_proxy value: {doc['video_age_proxy']}")

    # Count documents with new features
    query = {}
    if has_publish_hour:
        query['publish_hour'] = {'$exists': True}
    if has_video_age_proxy:
        query['video_age_proxy'] = {'$exists': True}

    if query:
        count_with_features = db.ml_features.count_documents(query)
        print(f"Documents with new features: {count_with_features}")

    print("\n=== SAMPLE DOCUMENT STRUCTURE ===")
    for key in sorted(doc.keys()):
        if key != '_id':
            value = doc[key]
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            print(f"{key}: {value}")

if __name__ == "__main__":
    check_mongodb_features()