// import Promise from 'bluebird'
import braintree from 'braintree';
import axios from 'axios';
import moment from 'moment-timezone';
import 'moment-round';
import _twilio from 'twilio';
import { clamp, sortBy, pickBy } from 'lodash';

import log from './utils/log.js';
import getTwilioPhone from './utils/getTwilioPhone';
import { formatTimeKey } from './utils/time';
import getPayout from './utils/getPayout';
import config from './config/config';

import { root } from './refs';

console.log(`root:`, root)

let fire = (ref) => {
  return new Promise((resolve, reject) => {
    ref.once(
      'value',
      (x) => {
        resolve(x.val());
      },
      reject
    );
  });
};

const twilio = new _twilio.RestClient(config.TWILIO_SID, config.TWILIO_TOKEN);

const delayInHours = Number(config.START_MATCHING_DELAY_HR) || 24;
// static configs
const default_tz = 'America/New_York';

let matchingProcesses = {
  // [jobid]: processId
};

const precondition = (condition, message = `Precondition not met`) => {
  if (!condition) {
    throw new Error(`PRECONDITION: ${message}`);
  }
};

console.log('local time:', new Date());

let allCandidates = {};

// helpers
const Duration = ({ hours = 0, minutes = 0, seconds = 0, hour, minute, second }) => {
  precondition(hour == null, `Don't use 'hour', use 'hours' #duration`);
 // precondition(minute == null, `Don't use 'minute', use 'minutes' #duration`);
 // precondition(second == null, `Don\'t use 'second', use 'seconds' #duration`);

  if (hours > 0) {
    return Duration({ minutes: minutes + 60 * hours, seconds });
  }
  if (minutes > 0) {
    return Duration({ seconds: seconds + 60 * minutes });
  }
  return seconds * 1000;
};

const hours = (num) => Duration({ hours: num });
const delayInMs = hours(delayInHours);

const getDistance = (coords_1, coords_2) => {
  // PAY ATTENTION: coords_1.latitude BUT coords_2.lat
  // this is because of legacy structure of data in FireBase
  const deg2rad = (deg) => deg * (Math.PI / 180); // degrees to radians
  const R = 6371; // Radius of the earth in km
  const distanceLatitude = deg2rad(coords_2.lat - coords_1.latitude);
  const distanceLongitude = deg2rad(coords_2.lng - coords_1.longitude);
  const a =
    Math.sin(distanceLatitude / 2) * Math.sin(distanceLatitude / 2) +
    Math.cos(deg2rad(coords_1.latitude)) *
      Math.cos(deg2rad(coords_2.lat)) *
      Math.sin(distanceLongitude / 2) *
      Math.sin(distanceLongitude / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  const distance = R * c; // Distance in km
  return distance || 0.1;
};

const sms = async ({ jobId, phone, text }) => {
  // TODO: regex for phone full validation
  if (!phone) {
    log.info(`${jobId} | ERROR sending sms: phone was not specified`);
    return true;
  }

  const twilioPhone = await getTwilioPhone(phone);
  await new Promise((yell, cry) => {
    twilio.messages.create(
      {
        body: text,
        to: phone,
        from: twilioPhone,
      },
      function(err, message) {
        if (err) {
          log.info(`${jobId} | ERROR sending sms:\n${JSON.stringify(err, null, '\t')}\n with text: ${text}`);
          cry(new Error(`Error sending SMS: ${err.message}`));
        } else {
          log.info(`${jobId} | SMS sent with text "${text}"`);
          yell();
        }
      }
    );
  });
};

const pushEvent = (jobId, event) => {
  root.child(`jobLog/${jobId}/events`).push({
    created: new Date().getTime(),
    ...event,
  });
};

// MATCHER

class FunctionQueue {
  constructor() {
    this.queue = [];
    this.running = false;
  }

  go() {
    if (this.queue.length) {
      this.running = true;
      this.queue.shift()(() => {
        this.go();
      });
    } else {
      this.running = false;
    }
  }

  add(func) {
    this.queue.push(func);
    if (!this.running) {
      this.go();
    }
  }
}
// Execution will automatically start once something is enqueued.
// This will help to avoid concurrent offers
const q = new FunctionQueue();

const offer = ({ jobId, processId, job, candidates }) => {
  q.add((cb) => {
    offerJob({ jobId, processId, job, candidates }, cb);
  });
};

const offerJob = async ({ jobId, processId, job, candidates }, next) => {
  log.debug(`Offering to candidates: ${candidates.join(', ')}`);

  if (candidates.length === 0) {
    log.info(`${jobId} | all candidates have offers: ${candidates.join(', ')}`);
    startMatchingWithDelay(jobId, 'all candidates have offers', processId);
    next();
    return null;
  }

  for (let candidateId of candidates) {
    // before offering, check if this candidate has offer atm
    const currentlyOfferedJob = await fire(root.child(`jobs/match/try/offered/candidates/${candidateId}`));
    // if so, then wait
    if (currentlyOfferedJob) {
      // don't wait, mark this candidate as missed and search for another one
      if (currentlyOfferedJob !== jobId) {
        //log.info(`${jobId} | can't be offered, because candidate ${candidateId} has offer atm`);
      }
      continue;
    } else {
      // else - offer this job
      const hoursBeforeShiftStart = Math.round((job.shift.start - Date.now()) / Duration({ hours: 1 }));
      let offerTimeout = clamp(hoursBeforeShiftStart - 2, 1, 30);

      log.info(
        //`${jobId} | OFFER JOB TO CANDIDATE ${candidateId}, it's around ${hoursBeforeShiftStart}hr before shift start, so timeout is ${offerTimeout} minutes`
      );

      await root.child(`jobs/match/try/offered/candidates/${candidateId}`).set(jobId);
      const profile = await fire(root.child(`users/profiles/${candidateId}`));
      pushEvent(jobId, {
        text: `Offer to ${profile.firstname} ${profile.lastname} (${candidateId}) for ${offerTimeout} minutes`,
        type: 'offered',
        id: candidateId,
      });

      const total_payout = getPayout.calculateRate(
        job.shift.start,
        job.shift.end,
        job.address.name,
        job.jobtypes,
        profile.isLegacyUser
      ).totalPayout;
      const rateString = `$${total_payout}`;

          // TEXT OF OFFER
      const offer = `
        Hi ${profile.firstname}!
        You have been matched with a shift at ${job.name} for ${job_get_momentjs({ job, for: 'start' }).format(
        'l'
      )} ${job_get_momentjs({ job, for: 'start' }).calendar()} till ${job_get_momentjs({
        job,
        for: 'end',
      }).calendar()} at ${job.address.name}.
        The total pay is approximately ${rateString}. ${
        job.parking
          ? job.parking.indexOf('No Parking') === -1 ? 'Parking options available. ' : 'NO parking available. '
          : ''
      }
        Respond with "accept" or "decline".
        For more shift details go to the Jobletics App
      `;


      await sms({
        jobId,
        phone: profile.phone,
        text: `SHIFT OFFER:\n${offer}`,
      });

      log.debug(`${jobId} | offered successfully`);

      // TODO Sure about this thing?
      setTimeout(() => {
        log.info(`${jobId} | OFFER EXPIRED: ${candidateId}, go on to the next candidate`);
        matchAgain(jobId, 'offerTimeout', processId);
      }, Duration({ minutes: offerTimeout }));

      next();

      break; // NOTE IMPORTANT - Stops next client from being offered too
    }
  }
};

const matchAgain = (jobId, reason, processId) => {
  // check if this matching process is the most recent one
  if (matchingProcesses[jobId] === processId) {
    matchJob(jobId, reason);
  } else {
    log.info(`${jobId} | STOP MATCHING INSTANCE ${processId}, as another one is circulating`);
  }
};

const uuid = () =>
  'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = (Math.random() * 16) | 0,
      v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });

const querystring = require('querystring');
const get_timezone_for_job = async (jobId, job) => {
  if (job.tz) {
    return job.tz;
  } else {
    const GOOGLE_API_KEY = 'AIzaSyAh-wIunj9NEWHJesT20Db_AG_JqieJjnI';
    const url = 'https://maps.googleapis.com/maps/api/timezone/json';
    const query = querystring.stringify({
      location: `${job.address.location.lat},${job.address.location.lng}`,
      timestamp: Math.round(job.shift.start / 1000),
      key: GOOGLE_API_KEY,
    });
    const response = await axios.get(`${url}?${query}`);
    const { status, timeZoneId } = await response.data;
    if (status === 'OK' && timeZoneId) {
      // TODO Make this not mutate here >_<
      job.tz = timeZoneId;
      await root.child(`jobs/items/${jobId}/tz`).set(timeZoneId);
      return timeZoneId;
    } else {
      log.info(
        `ERROR FETCHING TIMEZONE for job with params: location (${job.address.location.lat},${
          job.address.location.lng
        }) and timestamp (${job.shift.start})`
      );
      return default_tz;
    }
  }
};

const matchJob = async (jobId, reason) => {
  const processId = uuid();

  console.log(`reason:`, reason);

  matchingProcesses[jobId] = processId;
  log.info(`${jobId} | START MATCHING ${reason}`);

  pushEvent(jobId, {
    text: `Start matching (${reason})`,
    type: 'startMatching',
  });

  // validate job
  // check if job was accepted already
  const assignedCandidate = await fire(root.child(`jobs/jobs_in_progress/${jobId}`));
  log.info(`assignedCandidate:`, assignedCandidate);
  precondition(assignedCandidate == null, `job is already assigned to candidate ${assignedCandidate}`);

  // retrieve job item
  const job = await fire(root.child(`jobs/items/${jobId}`));

  // if this is a first round of matching
  if (reason === 'start') {
    // then start listening for accept and decline events
    // TODO Catch errors for this async fn
    root.child(`jobs/items/${jobId}/declined`).on('child_added', async (snap) => {
      const declinedCandidateId = snap.key();
      log.info(`${jobId} | JOB DECLINED: by ${declinedCandidateId}`);
      pushEvent(jobId, {
        text: `${declinedCandidateId} has declined offer`,
        type: 'declined',
        id: declinedCandidateId,
      });
      matchJob(jobId, 'onDecline');
      const declinedJob = await fire(root.child(`jobs/items/${jobId}`));

      if (declinedJob.group_uuid) {
        const duplicates = await fire(
          root
            .child(`jobs/items`)
            .orderByChild('group_uuid')
            .equalTo(declinedJob.group_uuid)
        );
        console.log(`duplicates:`, duplicates);
        for (let [declinedJobId, jobItem] of Object.entries(duplicates || {})) {
          await root
            .child(`jobs/items/${declinedJobId}/declined/${declinedCandidateId}`)
            .set(job.declined && job.declined[declinedCandidateId] ? job.declined[declinedCandidateId] : true);
        }
      }
    });
  }

  // TODO: put here job item validation to make sure the data is consistent
  // convert job start time into weekday and hour
  if (job.status !== 'await/match') {
    log.info(`${jobId} | STOP MATCHING: job status is not "await/match", it is ${job.status}`);
    pushEvent(jobId, {
      text: `Stop matching: job status is not "await/match", it is ${job.status}`,
      type: 'stopMatching',
    });
    return;
  }

  // notify admin, if job is close to overdue
  console.log(`job:`, job);
  const ms_before_start_shift = job.shift.start - Date.now();
  console.log('ms_before_start_shift:', ms_before_start_shift);

  if (ms_before_start_shift < 0) {
    // Start time already passed
    await root.child(`jobs/items/${jobId}`).update({
      status: 'cancelled',
      synched: null,
    });
    await root.child(`jobs/match/await/${jobId}`).remove();

    // TODO What is this exactly? -- DRAL
    fire(
      root
        .child(`jobs/match/try/offered/candidates`)
        .orderByValue()
        .equalTo(jobId)
    ).then((candidates) => {
      log.info('MISSED SHIFT?!', candidates);
      for (let candidateId of Object.keys(candidates || {})) {
        onMissedShift(candidateId, job, jobId);
      }
    });

    pushEvent(jobId, {
      text: 'Matcher has cancelled job',
      type: 'cancelJob',
    });

    log.info(`${jobId} | STOP MATCHING: job is outdated`);
    root.child('email').push({
      to: config.ADMIN_EMAIL,
      subject: 'Shift cancelled by matcher',
      body: `Shift cancelled by matcher, due to shift start time has already passed but it has not been accepted. Details: ${
        config.ADMIN_PANEL_URL
      }/job/${jobId}`,
    });
    return null;
  } else if (ms_before_start_shift < hours(0.5)) {
    if (!job.outdatedWarning_2) {
      log.info(`${jobId} | OUTDATED JOB: Less than 30 minutes before shift, but it still not accepted`);
      root.child('email').push(
        {
          to: config.ADMIN_EMAIL,
          subject: 'Shift gets outdated (30min)',
          body: `Less than 30 minutes before shift, but it still not accepted. Details: ${
            config.ADMIN_PANEL_URL
          }/job/${jobId}`,
        },
        () => {
          // make sure, this notification will not be sent more than once
          root.child(`jobs/items/${jobId}/outdatedWarning_2`).set(true);
        }
      );
    }
  } else if (ms_before_start_shift < hours(3)) {
    if (!job.outdatedWarning_1) {
      // if it's less than 3 hours before shft, then notify admin with email;
      log.info(`${jobId} | OUTDATED JOB: Less than 3 hours before shift, but it still not accepted`);
      log.info(`${jobId} | job data: ${job}`);
      root.child('email').push(
        {
          to: config.ADMIN_EMAIL,
          subject: 'Shift gets outdated (3hr)',
          body: `Less than 3 hours left but shift is not accepted. Details: ${config.ADMIN_PANEL_URL}/job/${jobId}`,
        },
        () => {
          // make sure, this notification will not be sent more than once
          root.child(`jobs/items/${jobId}/outdatedWarning_1`).set(true);
        }
      );
    }
  }

  // This script suppose to help us to migrate data structure for new version of app
  // TODO remove this logic, when all apps will be up to date
  if (reason === 'start') {
    root.child(`users/employers/activeJobs/${job.employerId}/${jobId}`).set(true);
  }

  if (job.name.indexOf('@') > -1) {
    // if job.name has `@` character, let's suppose, that job name is email
    const value = await root
      .child(`users/accounts`)
      .orderByChild('email')
      .equalTo(job.name);

    const candidateId = value ? Object.keys(value)[0] : null;
    // if candidateId has been found, then offer this job to this candidate.
    if (candidateId) {
      log.info(`${jobId} | CUSTOM MATCH: candidate ${candidateId}`);
      offer({
        jobId,
        processId,
        job,
        candidates: [candidateId],
      });
    } else {
      const tz = await get_timezone_for_job(jobId, job);
      await proceed(tz);
    }
  } else {
    const tz = await get_timezone_for_job(jobId, job);
    await proceed(tz);
  }

  // i know, this is ugly solution. It's done this way to make as small code changes as possible
  async function proceed(timezone) {
    console.log(`timezone:`, timezone);
    const startMoment = moment(job.shift.start)
      .tz(timezone)
      .round(15, 'minutes')
      .add(1, 'hour');
    // .add(1, 'hour') makes matcher think that job starts one hour later (#shiftLengthModifier)

    const startTime = {
      day: startMoment.format('dddd'),
      hour: startMoment.format('H'),
      minute: startMoment.format('m'),
      key: `${startMoment.hour}:${startMoment.minute}`,
    };
    const startTimeKey = formatTimeKey(startTime.hour, startTime.minute);

    const shiftDuration_timestamp = Math.abs(Number(job.shift.end) - Number(job.shift.start)) - hours(2);
    // ` - hours(2)` reduces shift length by 2 hours (#shiftLengthModifier)
    if (shiftDuration_timestamp > hours(24)) {
      log.info(
        `${jobId} | STOP MATCHING: shift duration is impossibly large: ${shiftDuration_timestamp /
          Duration({ hours: 1 })} hours`
      );
      pushEvent(jobId, {
        text: `Stop matching: shift duration is impossibly large: ${shiftDuration_timestamp /
          Duration({ hours: 1 })} hours`,
        type: 'stopMatch',
      });
      return;
    }

    const matchedCandidates = pickBy(allCandidates, (profile) => {
      return Boolean(
        profile.alt_schedule &&
          profile.alt_schedule[startTime.day] &&
          profile.alt_schedule[startTime.day][startTimeKey] === true
      );
    });
    log.info(`${jobId} | shift starts at ${startTime.day}/${startTimeKey}`);

    const candidate_results = Object.entries(matchedCandidates || {}).map(([candidateId, candidate]) => {
      // check if scheduled availability continuosly overlapping with shit time
      let missedInterval;
      const checkOverlap = () => {
        let completeMatchBySchedule = true;
        // this is for the case, if shift starts today, but ends next day
        const nextDay = startMoment.add(1, 'days').format('dddd');
        const firstDaySchedule = candidate.alt_schedule[startTime.day];
        const secondDaySchedule = candidate.alt_schedule[nextDay] || {};
        // how many 15-minutes steps is in this shift;
        let steps = shiftDuration_timestamp / Duration({ minutes: 15 });
        let sch = firstDaySchedule;
        let hour = Number(startTime.hour);
        let minute = Number(startTime.minute);
        while (steps > 0) {
          if (minute > 45) {
            // check next hour it's [any]hour:60minutes
            minute = 0;
            hour++;
          }
          if (hour > 23) {
            // check if it's 24th hour, which means "next day"
            sch = secondDaySchedule;
            hour = 0;
            minute = 0;
          }
          const timeKey = formatTimeKey(hour, minute);
          // if shift starts [e.g.] today and will be ended tomorow, then
          if (sch[timeKey]) {
            minute += 15;
            steps--;
          } else {
            // else stop this loop and mark this candidate as not matched by schedule
            missedInterval = timeKey;
            steps = 0;
            completeMatchBySchedule = false;
          }
        }
        return completeMatchBySchedule;
      }; // END checkOverlap

      const employer_review = candidate.employers && candidate.employers[job.employerId];

      const hasHomeAddress = Boolean(candidate.homeAddress);
      let candidateDontWantToWorkAgain = employer_review ? !employer_review.wouldWorkAgain : false;
      let employerDontWantToSeeAgain = employer_review ? !employer_review.wantSeeAgain : false;
      let preferred = employer_review && (employer_review.wouldWorkAgain || employer_review.wantSeeAgain);

      const busy_timeslots = Object.entries(candidate.jobs || {}).map(([acceptedJobId, acceptedJob]) => {
        return {
          start: Number(acceptedJob.start) - Duration({ hours: 1 }),
          end: Number(acceptedJob.end) + Duration({ hours: 1 }),
          Job_id: acceptedJobId,
          job: acceptedJob,
        };
      });

      const currently_busy_timeslot = busy_timeslots.find((busy_period) => {
        const now = Date.now();
        return busy_period.start < now && busy_period.end > now;
      });

      const offerStart = Number(job.shift.start);
      const offerEnd = Number(job.shift.end);
      const overlapping_shift = busy_timeslots.find((busy_period) => {
        // TODO Make sure this works !!!!
        return busy_period.start >= offerEnd && busy_period.end <= offerStart;
      });

      // jobtypes and certificates
      let missing_jobtypes = job.jobtypes.filter((jobType) => {
        return !candidate.jobTypes || !candidate.jobTypes[jobType];
      });

      let missing_certifications = Object.entries(job.certificates || {})
        .filter(([certification_name, required]) => {
          if (required) {
            return !candidate.certificates || !candidate.certificates[certification_name];
          } else {
            return true;
          }
        })
        .map((x) => x[0]);

      if (!hasHomeAddress) {
        return {
          candidateId,
          success: false,
          message: 'home address is absent',
        };
      } else if (overlapping_shift != null) {
        return {
          candidateId,
          success: false,
          message: `overlaps with ${overlapping_shift.job_id} (${moment(
            overlapping_shift.job.start
          ).calendar()} - ${moment(overlapping_shift.job.end).calendar()})`,
        };
      } else if (currently_busy_timeslot != null) {
        const message = `because is currently on shift: ${currently_busy_timeslot.job_id} (${moment(
          currently_busy_timeslot.job.start
        ).calendar()} - ${moment(currently_busy_timeslot.job.end).calendar()})`;
        return {
          candidateId,
          success: false,
          message: `is not available atm, ${message}`,
        };
      } else if (employerDontWantToSeeAgain) {
        return {
          candidateId,
          success: false,
          message: 'employer do not want to see again',
        };
      } else if (candidateDontWantToWorkAgain) {
        return {
          candidateId,
          success: false,
          message: 'candidate do not want to work here',
        };
      } else if (missing_jobtypes.length !== 0) {
        return {
          candidateId,
          success: false,
          message: `candidate doesnt have job types ${missing_jobtypes.join(', ')}`,
        };
      } else if (missing_certifications.length !== 0) {
        return {
          candidateId,
          success: false,
          message: `candidate doesnt have certificates ${missing_certifications.join(', ')}`,
        };
      } else if (!checkOverlap()) {
        // let this validator to be at the end, because it's the most expensive function among others
        return {
          candidateId,
          success: false,
          message: `availability is not completely overlapping shift time (${missedInterval})`,
        };
      } else {
        return {
          candidateId,
          success: true,
          candidate: candidate,
          preferred: preferred,
        };
      }
    });

    const matching_candidates = candidate_results.filter((x) => x.success === true);

    // job.declined => job.candidates_that_declined_this_job
    const candidates_not_declined = matching_candidates.filter(({ candidateId }) => {
      return !job.declined || !job.declined[candidateId];
    });

    // now we have array of matched candidates. Let's sort them by params and offer this job to the best candidate
    const candidates_with_penalty = candidates_not_declined
      .map((entry) => {
        const { candidate, preferred } = entry;
        let distance = getDistance(candidate.homeAddress.coords, job.address.location);
        const max = config.GEOFIRE_CANDIDATE_MAX_DISTANCE_KM;
        if (distance > max) {
          // matchingLog[candidate.candidateId] = `located further than ${max} km from the job`;
          return null;
        } else {
          const PENALTY_PER_KM = 0.1;
          const PENALTY_PER_MISSED_OFFER = 1;

          const missed_offers_amount = Number(candidate.missedOffersNumber) || 0;
          const penalty_for_missed_offers = missed_offers_amount * PENALTY_PER_MISSED_OFFER;
          const penalty_for_distance = distance * PENALTY_PER_KM;

          const penaltyPoints = penalty_for_missed_offers + penalty_for_distance;
          // matchingLog[candidate.candidateId] = `matched, ${Number(penaltyPoints).toFixed(
          //   3,
          // )} pp (${missedOffersNumber} missed offers; ${distance} km away;${candidate.preferred
          //   ? ' preferred candidate'
          //   : ''}`;
          return {
            ...entry,
            penaltyPoints: preferred ? 0 : penaltyPoints,
          };
        }
      })
      .filter(Boolean);

    if (!candidates_with_penalty[0]) {
      const marker = await fire(root.child(`jobLog/${jobId}/markers/candidateNotFound`));
      if (!marker) {
        root.child('email').push({
          to: config.ADMIN_EMAIL,
          //subject: `ATTN Candidate's not found for ${job.name}`,
          body: `Admin: ${config.ADMIN_PANEL_URL}/job/${jobId}`,
        });
        root.child(`jobLog/${jobId}/markers/candidateNotFound`).set(true);
      }
      console.log(`candidate_results:`, candidate_results);
      log.info(`${jobId} | try match again in ${config.MATCHER_TIMEOUT} minutes`);
      startMatchingWithDelay(jobId, 'candidate not found', processId);
      return;
    } else {
      const employer_prefered_order = await fire(root.child(`users/employers/jobletes/${job.employerId}/order`));

      const sortedCandidates = sortBy(candidates_with_penalty, (x) => {
        const prefered_index = employer_prefered_order && employer_prefered_order.indexOf(x.candidateId);
        return prefered_index ? -prefered_index : x.penaltyPoints;
      }).map((obj) => obj.candidateId);

      log.info(`${jobId} | MATCHED: \n ${JSON.stringify(candidate_results, null, '\t')}`);
      await offer({
        processId,
        jobId,
        job,
        candidates: sortedCandidates,
      });
    }
    // .catch(tryAgain => {
    //   log.info(`${jobId} | CANDIDATES NOT FOUND: \n${JSON.stringify(matchingLog, null, '\t')}`);
    //   root.child(`jobLog/${jobId}/match`).update(matchingLog);
    //
    //   if (tryAgain === true) {
    //     log.warn('USING TRY AGIAN :O');
    //     console.trace();
    //   } else {
    //   }
    // });
  }
};

const onMissedShift = async (candidateId, job, jobId) => {
  const now = new Date().getTime();
  await root.child(`analytics/candidates/${candidateId}/missed/${now}`).set(jobId);
  await root.child(`analytics/candidates/${candidateId}/availability/${now}`).set(false);
  await root.child(`jobs/candidates/joblete/${candidateId}`).remove();
  await root.child(`users/profiles/${candidateId}`).update({
    missed_shift: true,
    available: false,
    missedShiftTime: job.shift.start,
  });
  pushEvent(jobId, {
    text: `${candidateId} has missed offer`,
    type: 'missed',
    id: candidateId,
  });
  await root.child(`jobs/match/try/offered/candidates/${candidateId}`).remove();
  // notify candidate that he has missed an offer an is unavailable now
  const phone = await fire(root.child(`users/profiles/${candidateId}/phone`));
  const start_moment = job_get_momentjs({ job, for: 'start' });
  const text = `SHIFT MISSED OFFER:
  Another Joblete accepted the shift: ${job.name}. If you are no longer available ${start_moment.format(
    'dddd'
  )}//'s at ${start_moment.format('h:mm a')}, go to the Jobletics App and change your Availability.`;
  await sms({
    jobId,
    phone,
    text,
  });

  const missedOffersNumber_unsure = await fire(root.child(`users/profiles/${candidateId}/missedOffersNumber`));
  const missedOffersNumber = Number(missedOffersNumber_unsure) || 0;
  await root.child(`users/profiles/${candidateId}/missedOffersNumber`).set(missedOffersNumber + 1);
};

const startMatchingWithDelay = (jobId, reason, processId) => {
  setTimeout(() => {
    matchAgain(jobId, reason, processId);
  }, Duration({ minutes: config.MATCHER_TIMEOUT }));
};

const startMatcher = async () => {

  // Keep GLOBAL `allCandidates` variables in sync with... all candidates
  // TODO Move this to observable/more explicit?
  // TODO Use values$ or something else because this is aweful
  // @CONTINUES_RUNNING
  allCandidates = await fire(
    root
      .child('users/profiles')
      .orderByChild('type')
      .equalTo('candidate')
  );
  root
    .child('users/profiles')
    .orderByChild('type')
    .equalTo('candidate')
    .on('value', (snap) => {
      allCandidates = snap.val()
    });

  // First of all, let's check if there is outdated data in database

  // Delete current offers
  root.child(`jobs/match/try/offered`).remove();

  let job_that_can_have_errors = (job_fn) => {
    return async (...args) => {
      try {
        let promise = job_fn(...args);
        await promise;
      } catch (err) {
        console.error(`Error while doing job:`, err.stack);
      }
    };
  };

  // now start listening for new jobs
  // @CONTINUES_RUNNING
  root.child(`jobs/match/await`).on(
    'child_added',
    job_that_can_have_errors((snapshot) => {
      const jobId = snapshot.key();
      const jobStartTime = snapshot.val();
      const nowTime = new Date().getTime();

      log.info(`${jobId} | NEW JOB`);

      try {
        precondition(typeof jobStartTime === 'number', `JobStartTime is not a number (${jobStartTime})`);
      } catch (err) {
        root.child(`/jobs/match/await/${jobId}`).set(null);
        throw err;
      }

      // check if job matching needs to be delayed
      if (jobStartTime - nowTime > delayInMs) {
        log.info(`${jobId} | DELAY JOB: delayed due to start time is later than ${delayInHours} hours from now`);
        root.child(`jobs/match/await/${jobId}`).remove();
        root.child(`jobs/delay/${jobId}`).set(jobStartTime);
        // TODO Make sure /delay/ jobs get handled and removed the right way
        // TODO Push into delay (queue) and then add to await when active
      } else {
        matchJob(jobId, 'start').catch((error) => {
          console.log(`error:`, error);
          log.info(
            `${jobId} | POSSIBLE ERROR: matching was stopped without command to start again in ${
              config.MATCHER_TIMEOUT
            } minutes.`
          );
          pushEvent(jobId, {
            text: `Possible error: matching was stopped without command to start again in ${
              config.MATCHER_TIMEOUT
            } minutes.`,
            type: 'error',
          });
        });
      }
    })
  );

  // TODO -- Async function errors will be ignored by .on(...)
  // @CONTINUES_RUNNING
  root.child(`jobs/jobs_in_progress`).on('child_added', async (snapshot) => {
    const jobId = snapshot.key();
    const acceptedCandidateId = snapshot.val();
    const job = await fire(root.child(`jobs/items/${jobId}`));

    // if server has already seen this event for this job, then job.accepted_marker will be equal to true
    // this is necessary, because of firebase logic for `child_added` event:
    // https://www.firebase.com/docs/web/api/query/on.html
    if (!job.accepted_marker) {
      log.info(`${jobId} | JOB ACCEPTED: by ${acceptedCandidateId}`);
      await root.child(`jobs/items/${jobId}/accepted_marker`).set(true);
      await root.child(`jobs/match/waiting_for_confirmation/${jobId}`).set(job.shift.start);

      const candidate_profile = await fire(root.child(`users/profiles/${acceptedCandidateId}`));
      pushEvent(jobId, {
        text: `${candidate_profile.firstname} ${candidate_profile.lastname} (${acceptedCandidateId}) has accepted job`,
        type: 'accepted',
        id: acceptedCandidateId,
      });
      log.info(`${jobId} | JOB ACCEPTED by candidate ${acceptedCandidateId}`);

      // TODO Figure out what this is!
      const candidates = await fire(
        root
          .child(`jobs/match/try/offered/candidates`)
          .orderByValue()
          .equalTo(jobId)
      );
      for (let candidateId of Object.keys(candidates || {})) {
        await onMissedShift(candidateId, job, jobId);
      }
    }
  });

  var braintreeEnv =
    config.BRAINTREE_ENVIRONMENT === 'dev' ? braintree.Environment.Sandbox : braintree.Environment.Production;

  var gateway = braintree.connect({
    environment: braintreeEnv,
    merchantId: config.BRAINTREE_MERCHANT_ID,
    publicKey: config.BRAINTREE_PUBLIC_KEY,
    privateKey: config.BRAINTREE_PRIVATE_KEY,
  });
  const braintree_sale = ({ customerId, amount }) => {
    return new Promise((yell, cry) => {
      gateway.transaction.sale(
        {
          customerId: customerId,
          amount: amount,
          options: {
            submitForSettlement: true,
          },
        },
        function(err, result) {
          if (err) {
            cry(err);
          } else {
            yell(result);
          }
        }
      );
    });
  };

  // TODO Catch async errors
  // @CONTINUES_RUNNING
  root.child(`jobs/jobs_in_progress`).on('child_removed', async (snapshot) => {
    const jobId = snapshot.key();
    const candidateId = snapshot.val();

    const job = await fire(root.child(`jobs/items/${jobId}`));
    const { payout_done, charge, employer, status, candidatedata } = job;
    //const {payout_done, charge, employer, status, candidatedata, name} = job;

    if (payout_done) {
      return null;
    }

    if (status === 'cancelled') {
      log.info(`${jobId} | JOB CANCELLED IN PROGRESS`);
      pushEvent(jobId, {
        text: `Cancelled in progress`,
        type: 'cancelled',
      });
      const phone = await fire(root.child(`users/profiles/${candidateId}/phone`));
      sms({
        jobId,
        phone,
        text: `Hi ${
          candidatedata.name
        }, one of your current jobs has been cancelled. Go into the Jobletics App to verify the shift is cancelled.`,
        //text: `Hi ${candidatedata.name}, your current job ${name} has been cancelled. Go into the Jobletics App to verify the shift is cancelled.`,
      });
    } else {
      log.info(`${jobId} | JOB FINISHED: by candidate ${candidateId}`);
    }

    if (!charge) {
      log.info(`${jobId} | NOTHING TO CHARGE: "charge" property is absent`);
    } else {
      const linkedAccounts = await fire(root.child('users/linkedAccounts'));
      let id = employer;
      Object.entries(linkedAccounts).forEach(([uid, { items }]) => {
        items.forEach((linkedAccount) => {
          if (linkedAccount.fbKey === employer && linkedAccount.withPayments && linkedAccount.confirmed) {
            id = uid;
          }
        });
      });

      await braintree_sale({ customerId: id.replace(/-/g, '_'), amount: charge });
      log.info(`BRAINTREE PAYMENT DONE`);
      await root.child(`jobs/items/${jobId}/payout_done`).set(true);
    }
  });
};

const confirmationChecker = (app) => {
  // TODO Catch async error
  // @CONTINUES_RUNNING
  setInterval(async () => {
    const jobs = await fire(root.child(`jobs/match/waiting_for_confirmation`));
    const nowTime = new Date().getTime();
    const checkPeriod = Duration({ minutes: config.CHECK_CANDIDATE_PERIOD });
    const checkDelay = Duration({ minutes: config.CHECK_CANDIDATE_DELAY });

    for (let [jobId, startTime] of Object.entries(jobs || {})) {
      const timeLeft = Number(startTime) - nowTime;
      if (timeLeft < checkPeriod - checkDelay) {
        // TODO
        // This is nice, this is how I would want everything to work hehehe
        // Buuuuuut I think there is a better platform for this than firebase :-/
        const sendEmail = (name, candidateId) => {
          root.child('email').push({
            to: config.ADMIN_EMAIL,
            subject: `${name} HAS NOT CONFIRMED SHIFT ATTENDANCE`,
            body: `${name} hasn''t confirmed that they are still going to shift. \nDetails: ${
              config.ADMIN_PANEL_URL
            }/job/${jobId} \nCandidate profile: ${config.ADMIN_PANEL_URL}/user/${candidateId}`,
          });
        };

        await root.child(`jobs/match/waiting_for_confirmation/${jobId}`).remove();
        const candidateId = await fire(root.child(`jobs/items/${jobId}/candidate`));
        if (candidateId) {
          const profile = await fire(root.child(`users/profiles/${candidateId}`));
          if (profile) {
            sendEmail(`${profile.firstname} ${profile.lastname}`, candidateId);
          } else {
            sendEmail('unknown', candidateId);
          }
        } else {
          sendEmail('unknown', candidateId);
        }

        pushEvent(jobId, {
          text: `Candidate hasn''t confirmed shift attendance`,
          type: 'notConfirmed',
        });
      } else if (timeLeft < checkPeriod) {
        const job = await fire(root.child(`jobs/items/${jobId}`));
        if (!job.candidateConfirmationSmsSent) {
          const candidateId = job.candidate;
          const candidate = await fire(root.child(`users/profiles/${candidateId}`));
          await sms({
            jobId,
            phone: candidate.phone,
            text: `SHIFT CONFIRMATION:
            Hi ${candidate.firstname}, this is a confirmation that your scheduled shift at ${
              job.name
            } starts at ${job_get_momentjs({ job, for: 'start' }).format(
              'h:mm a'
            )}. Please confirm you will be on time by replying "confirm" within 30 minutes`,
          });
          await root.child(`jobs/items/${jobId}/candidateConfirmationSmsSent`).set(true);
        }
      }
    }
  }, Duration({ minutes: 5 })); // 5 minutes
};

// Get a momentjs object
const job_get_momentjs = ({ job, for: forName }) => {
  const timezone = job.tz || default_tz;
  if (forName === 'start') {
    return moment(job.shift.start).tz(timezone);
  } else if (forName === 'end') {
    return moment(job.shift.end).tz(timezone);
  } else {
    throw new Error(`job_get_momentjs only works with for=start or for=end, not for='${forName}'`);
  }
};

const shiftReminder = (app) => {
  log.info('SHIFT REMINDER STARTING !!!');
  // TODO setInterval ignores async errors
  // @CONTINUES_RUNNING
  setInterval(async () => {
    const jobs = await fire(root.child(`jobs/match/waiting_for_confirmation`));
    const nowTime = Number(new Date().getTime());
    const checkPeriod = Duration({ minutes: config.CHECK_CANDIDATE_REMINDER });

    for (let [jobId, startTime] of Object.entries(jobs || {})) {
      // TODO Is this `Number` necessary? I'd rather error if it is not a number
      if (typeof startTime !== 'number') {
        log.warn(`StartTime is not a number?! (@shiftReminder):`, jobId, startTime);
      }

      const timeLeft = Number(startTime) - nowTime;
      if (timeLeft < checkPeriod) {
        const job = await fire(root.child(`jobs/items/${jobId}`));
        if (!job.candidateConfirmationSmsSent) {
          const candidateId = job.candidate;
          const candidate = await fire(root.child(`users/profiles/${candidateId}`));
          await sms({
            jobId,
            phone: candidate.phone,
            text: `SHIFT REMINDER:
            Hi ${candidate.firstname}, this is a reminder your shift at ${job.name} on ${job_get_momentjs({
              job,
              for: 'start',
            }).format('l')} starts at ${job_get_momentjs({ job, for: 'start' }).format(
              'h:mm a'
            )}. Please make sure you arrive on time, are in the right uniform, and have appropriate transportation. For shift details, goto your Jobletics App. Have fun!`,
          });
        }
      }
    }
  }, Duration({ minutes: 1405 })); // 1440 minutes
  // TODO WHy is this such a weird number? (23,41667 hours) -- DRAL
};

/*
Currently only handles "cancelJob" events sent from phones
*/
const listenFirebaseEvents = () => {
  // TODO .on ignores async errors
  root.child('events').on('child_added', async (snap) => {
    const eventId = snap.key();
    const event = snap.val();

    console.log('New event:', event.eventName, event.data);

    if (event.eventName === 'cancelJob') {
      // TODO Penalty comes from the event? Why is that?
      const { jobId, penalty, by } = event.data;

      // TODO Make sure these two actually get awaited
      if (penalty) {
        await root.child(`jobs/items/${jobId}`).update({
          payout: 0,
          charge: 20,
          status: 'cancelled',
          synched: null,
        });
      } else {
        await root.child(`jobs/items/${jobId}`).update({
          status: 'cancelled',
          synched: null,
        });
      }

      const candidateId = await fire(root.child(`jobs/jobs_in_progress/${jobId}`));
      if (candidateId) {
        await root.child(`users/profiles/${candidateId}/available`).set(true);
        await root.child(`users/profiles/${candidateId}/jobs/${jobId}`).remove();
        await root.child(`analytics/candidates/${candidateId}/availability/${new Date().getTime()}`).set(true);
        await root.child(`jobs/jobs_in_progress/${jobId}`).remove();
      }
      await root.child(`jobs/match/waiting_for_confirmation/${jobId}`).remove();

      await root.child(`jobs/match/await/${jobId}`).remove();
      const job = await fire(root.child(`jobs/items/${jobId}`));
      const candidates = await fire(
        root
          .child(`jobs/match/try/offered/candidates`)
          .orderByValue()
          .equalTo(jobId)
      );

      for (let candidateId of Object.keys(candidates || {})) {
        await onMissedShift(candidateId, job, jobId);
      }
      pushEvent(jobId, {
        type: 'cancelled',
        text: `${by} has cancelled job`,
      });
      await root.child(`events/${eventId}`).remove();
    }
  });
};

// check if there are delayed jobs each hour
const checkDelayedJobs = async () => {
  log.info('DELAYED JOBS HOURLY CHECK');
  const delayedJobs = await fire(root.child(`jobs/delay`));

  if (delayedJobs) {
    log.info(`Currently delayed jobs: ${JSON.stringify(delayedJobs, null, '\t')}`);
    const nowTime = new Date().getTime();

    for (let [jobId, startTime] of Object.entries(delayedJobs || {})) {
      if (startTime - nowTime < delayInMs) {
        log.info(`${jobId} | Job is less than ${delayInHours} from now. Checking status of this shift.`);
        const status = await fire(root.child(`jobs/items/${jobId}/status`));
        log.info(`${jobId} | Status is ${status}`);

        if (status === 'await/match') {
          await root.child(`jobs/delay/${jobId}`).remove();
          await root.child(`jobs/match/await/${jobId}`).set(true);
        } else {
          await root.child(`jobs/delay/${jobId}`).remove();
          log.info(`${jobId} | Status is not "await/match", removing shift from delay node without matching`);
        }
      }
    }
  } else {
    log.info('Currently there are no delayed jobs');
  }
};

const auth_firebase = () => {
  return new Promise((yell, cry) => {
    console.log('Authing firebase', root.authWithCustomToken)
    root.authWithCustomToken(config.FIREBASE_SECRET, (error, authData) => {
      console.log('Done authing firebase')
      if (error) {
        cry(error);
      } else {
        yell(authData);
      }
    });
  });
};

export const startWorkers = async () => {
  if (config.MATCHER_ONLINE === 'false' || config.MATCHER_ONLINE === false) {
    log.info('MATCHER IS OFFLINE');
  } else {
    await auth_firebase();

    startMatcher();

    checkDelayedJobs();
    // run each hour and on server start
    setInterval(checkDelayedJobs, hours(1));

    confirmationChecker();
    shiftReminder();
    listenFirebaseEvents();

    log.info('Started job matching workers!');
  }
};

export default {
  startWorkers,
};
