CREATE TABLE hospital (
    id int primary key,
    hospital_name VARCHAR(100),
    location VARCHAR(255) NOT NULL,
    contact_details VARCHAR(50),
    rating float,
    treatments_available VARCHAR(255),
    email VARCHAR(100)
);

INSERT INTO hospital (id,hospital_name, location, contact_details, rating, treatments_available, email) 
VALUES 
(123,'chennai_aravind', 'Poonamalle High Rd, Poonamallee, Chennai, Tamil Nadu', '04440956100', 4.0, 'Cataract, Childrens Eye Care, Cornea, General Ophthalmology, Glaucoma, Low Vision & Vision Rehabilitation, Neuro Opthalmology, Orbit, Oculoplasty, Prosthetics & Oncology, Retina & Vitreous, Uvea', 'info.chennai@aravind.org'),
(124,'madurai_aravind', 'Kuruvikaran Salai, Anna Nagar, Shenoy Nagar, Madurai, Tamil Nadu 625020', '(0452) 435 6105', 4.1, 'Cataract, Corneal, Refractive, Glaucoma Care, Low Vision, Vision Rehabilitation, Pediatric Ophthalmology, Ocular Oncology, Retinal Diseases, Retinal Detachment, Diabetic Retinopathy, Macular Degeneration', 'patientfeedback@aravind.org, patientcare@aravind.org'),
(125,'coimatore_aravind', 'Avinashi Road, Coimbatore 641 014, Tamil Nadu, India', '(0422) 436 0400', 4.5, 'Eye Exams, Diagnosis, Treatment of Eye Diseases, Cataract Removal, Glaucoma Surgery, Retinal Detachment Repair', 'cbe.info@aravind.org'),
(126,'tirunelveli_aravind', 'S.N. High Road, Tirunelveli Jn. 627 001, Tamil Nadu, India', '(0462) 435 6100', 4.5, 'Cataract Surgery, Corneal Transplantation, Glaucoma Treatment, Pediatric Ophthalmology, Vitrectomy, Lasik Surgery, Low Vision, Vision Rehabilitation, Neuro Ophthalmology, Retina and Vitreous', 'tvl.aravind@aravind.org'),
(127,'tirupati_aravind', 'Alipiri to Zoo park Road, Beside NCC Campus, Tirupati – 517 505, Andhra Pradesh', '(0877) 2502100', 4.5, 'Cataract Surgery, Glaucoma Treatment, Cornea Transplant, LASIK, Children''s Eye Care, Low Vision, Retina/Vitreous Diseases, Teleophthalmology', 'tpt.patientcare@aravind.org');

CREATE TABLE emp (
    name VARCHAR(100) NOT NULL,
    employee_id VARCHAR(30) PRIMARY KEY,
    employee_contact VARCHAR(15),
    emp_branch VARCHAR(50),
    location VARCHAR(255),
    email_id VARCHAR(100)
);

INSERT INTO emp (name, employee_id, employee_contact, emp_branch, location, email_id) 
VALUES 
('kamesh', '122311510204', '7654321098', 'chennai', 'chennai', 'kamesh@aravid.org'),
('yasin shaif', '122311520201', '8765432109', 'tirunelveli', 'thirunelveli', 'yasinsharif@aravind.org'),
('dhana sekar', '122311530203', '6543210987', 'coimatore', 'kanchipuram', 'dhanasekar@aravind.org'),
('lokanthan monika', '122311540232', '9876543201', 'tirupati', 'chittoor', 'lokanathanmonika@aravid.org'),
('lokanthan dimpul', '122311550241', '9871234560', 'madurai', 'chittoor', 'lokanathandimpul@aravid.org');


ALTER TABLE emp ADD CONSTRAINT UNIQUE (employee_id);
ALTER TABLE hospital ADD CONSTRAINT UNIQUE (hospital_name);


CREATE TABLE org (
    emp_id VARCHAR(30),
    hospital_name VARCHAR(100),
    branch VARCHAR(50),
    emp_contact VARCHAR(15),
    PRIMARY KEY (emp_id, hospital_name),
    CONSTRAINT fk_emp_id FOREIGN KEY (emp_id) REFERENCES emp(employee_id),
    CONSTRAINT fk_hospital_name FOREIGN KEY (hospital_name) REFERENCES hospital(hospital_name)
);


INSERT INTO org (emp_id, hospital_name, branch, emp_contact)
SELECT 
    employee_id, 
    CASE emp_branch
        WHEN 'chennai' THEN 'chennai_aravind'
        WHEN 'madurai' THEN 'madurai_aravind'
        WHEN 'coimatore' THEN 'coimatore_aravind'
        WHEN 'tirunelveli' THEN 'tirunelveli_aravind'
        WHEN 'tirupati' THEN 'tirupati_aravind'
    END,
    emp_branch,
    employee_contact
FROM emp;

CREATE TABLE role (
    emp_id VARCHAR(30) PRIMARY KEY,
    salary DECIMAL(10, 2),
    reporting_to VARCHAR(100),
    experience INT,
    DOJ DATE,
    emp_name VARCHAR(100),
    FOREIGN KEY (emp_id) REFERENCES emp(employee_id)
);


INSERT INTO role (emp_id, salary, reporting_to, experience, DOJ, emp_name)
VALUES
('122311510204', 100000, 'Sathya', 5, '2019-07-09', 'Kamesh'),
('122311520201', 75000, 'Sathya', 5, '2019-07-09', 'Yasin Shaif'),
('122311530203', 50000, 'Sathya', 4, '2020-08-23', 'Dhana Sekar'),
('122311540232', 100000, 'Kamesh', 3, '2021-09-07', 'Lokanthan Monika');   